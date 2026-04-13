import numpy as np
import torch
import torch.nn as nn

from bmt.dataset import constants
from bmt.models.layers import polyline_encoder, common_layers, position_encoding_utils
# from torch.nn.modules.transformer import TransformerEncoderLayer as NativeTransformerEncoderLayer
from bmt.models.layers.encoder_layer import TransformerEncoderLayer  # as NativeTransformerEncoderLayer
from bmt.models.ops.collapse_time import collapse_time
from bmt.utils import rotate, unwrap


def mode_agent_id(agent_id, max_agents, fill_negative_1=False):
    # As most of the "modeled agents" are in the first few agents, we want to remap those useless agents to latter
    # positions.
    agent_id = agent_id.clone()
    if fill_negative_1:
        agent_id[torch.logical_or(agent_id >= max_agents, agent_id < 0)] = -1
    else:
        agent_id[torch.logical_or(agent_id >= max_agents, agent_id < 0)] = max_agents - 1
    return agent_id


def find_last_valid(array, mask):
    assert mask.ndim + 1 == array.ndim
    assert mask.shape == array.shape[:-1]
    assert array.ndim == 4
    B, T, N, D = array.shape
    indices = mask * torch.arange(T, device=mask.device).reshape(1, T, 1).expand(*mask.shape)
    indices = indices.argmax(1, keepdims=True).unsqueeze(-1).expand(B, 1, N, D)
    ret = torch.gather(array, index=indices, dim=1)  # [B, 1, N, D]
    ret[~mask.any(1, keepdims=True)] = 0
    return ret


def pairwise_mask(mask):
    """
    input mask is in shape (B, N), we need to prepare a pairwise mask in shape (B, N, N).
    It's not correct to naively expand the mask. We need to maintain the symmetry of the mask.
    """
    B, N = mask.shape
    mask = mask.unsqueeze(1).expand(B, N, N)
    mask = mask & mask.transpose(1, 2)
    return mask


def pairwise_relative_diff(positions):
    """
    Compute pairwise relative diffs for a batch of objects.
    For the ouput [b, i, j, :], it means the relative differences of [b, j] - [b, i],
    which is the pos of j in i's coordinate system.

    Parameters:
    - positions: A PyTorch tensor of shape (B, N, 2)

    Returns:
    - A PyTorch tensor of shape (B, N, N, 2) containing pairwise relative positions.
    """

    # Expand dimensions to get tensors of shapes (B, N, 1, ...) and (B, 1, N, ...)
    positions_expanded_a = positions.unsqueeze(2)  # Shape: (B, N, 1, ...)
    positions_expanded_b = positions.unsqueeze(1)  # Shape: (B, 1, N, ...)

    # Compute the pairwise relative positions by subtraction
    relative_positions = positions_expanded_b - positions_expanded_a  # Shape: (B, N, N, ...)

    return relative_positions


def compute_relation(pos, heading, mask, hidden_dim, knn=128):
    """
    Compute the relation encoding for the transformer encoder.
    """
    assert heading.ndim == 2
    assert pos.ndim == 3
    pairwise_heading = pairwise_relative_diff(heading)
    heading_fill_0_mask = pairwise_mask(heading == constants.HEADING_PLACEHOLDER)
    pairwise_heading[heading_fill_0_mask] = 0

    rel_pos = pairwise_relative_diff(pos[..., :2])

    B, N = heading.shape
    # i's local coordinate's y-axis (the heading) in the global coordinate
    i_local_y_wrt_global = heading.reshape(B, N, 1).expand(B, N, N)
    i_local_x_wrt_global = i_local_y_wrt_global - np.pi / 2
    rotated_pos = rotate(rel_pos[..., 0], rel_pos[..., 1], angle=-i_local_x_wrt_global)

    mask = pairwise_mask(mask)

    THRESHOLD = 100
    dist = rel_pos.norm(dim=-1)
    dist_mask = dist < THRESHOLD
    rel_mask = torch.logical_and(mask, dist_mask)

    indices = None
    if knn:
        dist = dist.masked_fill(~mask, float("+inf"))
        indices = dist.argsort(dim=-1)[..., :knn]

        rotated_pos = torch.gather(rotated_pos, dim=-2, index=indices.unsqueeze(-1).expand(-1, -1, -1, 2))
        pairwise_heading = torch.gather(pairwise_heading, dim=-1, index=indices)
        rel_mask = torch.gather(rel_mask, dim=-1, index=indices)

    pos_pe = position_encoding_utils.gen_sineembed_for_relation(
        rotated_pos[rel_mask], pairwise_heading[rel_mask], hidden_dim=hidden_dim
    )
    pos_pe = unwrap(pos_pe, rel_mask)
    return pos_pe, rel_mask, indices


class SceneEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # TODO: Pass this from config or datasource
        SCENE_INPUT_TIME_STEPS = 11
        self.history_steps = SCENE_INPUT_TIME_STEPS
        self.config = config
        self.d_model = self.config.MODEL.D_MODEL
        self.num_layers = self.config.MODEL.NUM_ATTN_LAYERS

        self.map_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=constants.MAP_FEATURE_STATE_DIM,
            hidden_dim=64,
            num_layers=2,
            num_pre_layers=1,
            out_channels=self.d_model
        )
        self.agent_mlps = common_layers.build_mlps(
            c_in=constants.AGENT_STATE_DIM * SCENE_INPUT_TIME_STEPS,
            mlp_channels=[self.d_model] * 3,
            ret_before_act=True,
        )
        self.light_mlps = common_layers.build_mlps(
            c_in=constants.TRAFFIC_LIGHT_STATE_DIM * SCENE_INPUT_TIME_STEPS,
            mlp_channels=[self.d_model] * 3,
            ret_before_act=True,
        )

        # self.separate_pe = self.config.MODEL.get('SEPARATE_PE', False)

        dropout = self.config.MODEL.DROPOUT_OF_ATTN
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        self_attn_layers = []
        # transformer_d_model = self.d_model * 2 if self.separate_pe else self.d_model
        for _ in range(self.num_layers):
            self_attn_layers.append(
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.num_heads,
                    dim_feedforward=self.d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    pre_projection=self.config.MODEL.get('PRE_PROJECTION', False),
                    relative_pe=self.config.MODEL.get('RELATIVE_PE', False),
                )
            )

        self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.agent_pe = nn.Embedding(self.config.PREPROCESSING.MAX_AGENTS, self.d_model)

        self.out = common_layers.build_mlps(
            c_in=self.d_model,
            mlp_channels=[self.d_model],
            ret_before_act=True,
        )

        self.relative_pe = self.config.MODEL.get('RELATIVE_PE', False)

        self.add_pe_for_static_features = self.config.MODEL.get('ADD_PE_FOR_STATIC_FEATURE', False)
        if self.add_pe_for_static_features:
            self.type_pe = common_layers.Tokenizer(num_actions=constants.NUM_TYPES, d_model=self.d_model)

    def forward(self, input_dict):

        # ===== Get shape =====
        B, T, N, D_agent = input_dict["encoder/agent_feature"].shape
        _, M, num_vector, D_vector = input_dict["encoder/map_feature"].shape
        _, _, L, D_light = input_dict["encoder/traffic_light_feature"].shape
        in_evaluation = input_dict["in_evaluation"][0].item()

        # ===== Embed agent feature =====
        agent_feature = input_dict["encoder/agent_feature"]
        agent_valid_mask = input_dict["encoder/agent_valid_mask"]
        agent_position = input_dict["encoder/agent_position"]
        agent_heading = input_dict["encoder/agent_heading"]
        agent_id = input_dict["encoder/agent_id"]
        assert agent_feature.shape[:3] == agent_position.shape[:3] == agent_valid_mask.shape[:3]
        agent_feature = (agent_feature[:, :self.history_steps] * agent_valid_mask[:, :self.history_steps, ..., None])
        agent_feature = collapse_time(agent_feature)
        agent_token = self.agent_mlps(agent_feature)  # (B, N, D)

        if in_evaluation:
            # Exempt filtering for maximum number of agents, so agent_id might be out of bound.
            agent_id = mode_agent_id(agent_id, self.config.PREPROCESSING.MAX_AGENTS)
            # Exempt filtering for maximum number of agents, so agent_id might be out of bound.
            modeled_agent_id = mode_agent_id(
                input_dict["encoder/modeled_agent_id"], self.config.PREPROCESSING.MAX_AGENTS
            )
        else:
            modeled_agent_id = input_dict["encoder/modeled_agent_id"]

        if self.config.MODEL.RANDOMIZE_AGENT_ID:
            weights = torch.ones(self.config.PREPROCESSING.MAX_AGENTS).expand(B, -1)
            if N > self.config.PREPROCESSING.MAX_AGENTS:
                new_encoder_agent_id = torch.full_like(agent_id, -1)
                num_samples = self.config.PREPROCESSING.MAX_AGENTS
                new_encoder_agent_id[:, :num_samples] = torch.multinomial(
                    weights, num_samples=num_samples, replacement=False
                ).to(agent_id)
                assert (agent_id[:, num_samples:] == self.config.PREPROCESSING.MAX_AGENTS - 1).all()
            else:
                num_samples = N
                new_encoder_agent_id = torch.multinomial(
                    weights, num_samples=num_samples, replacement=False
                ).to(agent_id)
                new_encoder_agent_id[agent_id == -1] = N
            input_dict["encoder/randomized_agent_id"] = new_encoder_agent_id
            agent_id = new_encoder_agent_id

            modeled_agent_mask = torch.logical_or(modeled_agent_id == -1, modeled_agent_id >= N)
            modeled_agent_id[modeled_agent_mask] = N - 1  # Quick workaround
            new_modeled_agent_id = torch.gather(new_encoder_agent_id, dim=1, index=modeled_agent_id)
            # new_modeled_agent_id[modeled_agent_mask] = N - 1
            input_dict["encoder/randomized_modeled_agent_id"] = new_modeled_agent_id
            modeled_agent_id = new_modeled_agent_id

        else:
            raise ValueError("Please turn on MODEL.RANDOMIZE_AGENT_ID=True")

        agent_id = mode_agent_id(agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=False)
        modeled_agent_id = mode_agent_id(modeled_agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=False)

        agent_pe = self.agent_pe(agent_id)  # (B, N, D)
        agent_token += agent_pe
        assert agent_token.shape == (B, N, self.d_model)

        if self.add_pe_for_static_features:
            type_pe = self.type_pe(input_dict["encoder/agent_type"])
            agent_token += type_pe  # [:, None]

        # ===== Embed map feature =====
        map_feature = input_dict["encoder/map_feature"]
        map_valid_mask = input_dict["encoder/map_feature_valid_mask"]
        map_position = input_dict["encoder/map_position"]
        map_heading = input_dict["encoder/map_heading"]
        map_token_valid_mask = input_dict["encoder/map_valid_mask"]
        map_token = self.map_polyline_encoder(map_feature, map_valid_mask)
        assert map_token.shape == (B, M, self.d_model)

        # ===== Embed traffic light =====
        traffic_light_feature = input_dict["encoder/traffic_light_feature"]
        traffic_light_position = input_dict["encoder/traffic_light_position"]
        traffic_light_heading = input_dict["encoder/traffic_light_heading"]
        traffic_light_valid_mask = input_dict["encoder/traffic_light_valid_mask"]
        if L != 0:
            traffic_light_feature = (
                traffic_light_feature[:, :self.history_steps] *
                traffic_light_valid_mask[:, :self.history_steps, ..., None]
            )
            traffic_light_feature = collapse_time(traffic_light_feature)
            traffic_light_token = self.light_mlps(traffic_light_feature)
        else:
            traffic_light_token = traffic_light_feature.new_zeros([B, L, self.d_model])
        assert traffic_light_token.shape == (B, L, self.d_model)

        # ===== Call transformer layers =====
        x = torch.concatenate([map_token, agent_token, traffic_light_token], dim=1)

        # ======== changes for including language embedding into scenario encoding features
        if self.config.LANGUAGE_CONDITION:
            if 'decoder/prompt_embedding' not in input_dict:
                print("PROMPT EMBED NOT FOUND")
                raise ()
            else:
                prompt_embedding = input_dict['decoder/prompt_embedding']
                print("x.shape", x.shape, "embeding.shape", prompt_embedding.shape)
                expanded_embedding = prompt_embedding.unsqueeze(-1).repeat(
                    1, 1, 256
                )  # Repeating to shape (6, 512, 256)
                expanded_embedding_mask = torch.ones((expanded_embedding.shape[0], expanded_embedding.shape[1]))
                x = torch.cat([x, expanded_embedding], dim=1)
        # ========

        x_pos = torch.concatenate(
            [
                map_position,
                find_last_valid(agent_position[:, :self.history_steps], agent_valid_mask[:, :self.history_steps])[:, 0],
                traffic_light_position
            ],
            dim=1
        )

        x_mask = torch.concatenate(
            [
                map_token_valid_mask, agent_valid_mask[:, :self.history_steps].any(dim=1),
                traffic_light_valid_mask[:, :self.history_steps].any(dim=1)
            ],
            dim=1
        )
        assert torch.all(x_mask.sum(dim=-1) > 0)

        if self.relative_pe:
            x_heading = torch.concatenate(
                [
                    map_heading,
                    find_last_valid(
                        agent_heading[:, :self.history_steps, ..., None], agent_valid_mask[:, :self.history_steps]
                    )[:, 0, :, 0], traffic_light_heading
                ],
                dim=1
            )
            relation, rel_mask, indices = compute_relation(
                pos=x_pos,
                heading=x_heading,
                mask=x_mask,
                hidden_dim=self.d_model,
                knn=self.config.MODEL.get('KNN', 128)
            )
            pos_embedding = None

            # To speed up:
            # assert rel_mask.ndim == 3
            # rel_mask = rel_mask.view(B, 1, rel_mask.shape[1], rel_mask.shape[2])\
            #     .expand(-1, self.num_heads, -1, -1)\
            #     .reshape(B * self.num_heads, rel_mask.shape[1], rel_mask.shape[2])
        else:
            relation = None
            pos_embedding = position_encoding_utils.gen_sineembed_for_position(x_pos[..., 0:2], hidden_dim=self.d_model)

        for k in range(len(self.self_attn_layers)):
            # inp = self._add_pe(x, pos_embedding)
            x = self.self_attn_layers[k](
                tgt=x,
                pos=pos_embedding,
                tgt_key_padding_mask=~x_mask,
                relation=relation,
                relation_mask=rel_mask,
                relation_indices=indices,
            )

        # x = torch.cat([x, pos_embedding], dim=-1)
        x = self.out(x.reshape(-1, x.shape[-1])).reshape(list(x.shape[:-1]) + [self.d_model])

        if pos_embedding is not None:
            x = x + pos_embedding

        input_dict["encoder/scenario_token"] = x
        if self.relative_pe:
            input_dict["encoder/scenario_position"] = x_pos
            input_dict["encoder/scenario_heading"] = x_heading
        input_dict["encoder/scenario_valid_mask"] = x_mask

        input_dict["encoder/modeled_agent_pe"] = self.agent_pe(modeled_agent_id)
        if self.add_pe_for_static_features:
            input_dict["encoder/modeled_agent_type_pe"] = self.type_pe(input_dict["encoder/modeled_agent_type"])
        return input_dict


if __name__ == '__main__':
    from bmt.utils import debug_tools

    config = debug_tools.get_debug_config()
    model = SceneEncoder(config)
    input_dict = debug_tools.get_debug_data()
    out = model(input_dict)
    print(out)
