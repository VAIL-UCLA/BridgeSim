import torch
import torch.nn as nn

from bmt.dataset import constants
from bmt.models import relation
from bmt.models.layers import polyline_encoder, common_layers, fourier_embedding
from bmt.models.layers.gpt_encoder_layer import SelfAttTransformerEncoder, SelfAttTransformerEncoderLayer
from bmt.models.motion_decoder_gpt import get_edge_info
from bmt.models.ops.collapse_time import collapse_time
from bmt.models.scene_encoder import find_last_valid, mode_agent_id
from bmt.utils import utils


class SceneEncoderGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO: Pass this from config or datasource
        SCENE_INPUT_TIME_STEPS = 11
        self.history_steps = SCENE_INPUT_TIME_STEPS
        self.config = config
        self.d_model = self.config.MODEL.D_MODEL
        self.num_layers = self.config.MODEL.NUM_ATTN_LAYERS
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD

        dropout = self.config.MODEL.DROPOUT

        is_v7 = self.config.MODEL.IS_V7
        self.is_v7 = is_v7

        self.map_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=constants.MAP_FEATURE_STATE_DIM,
            hidden_dim=64,
            num_layers=2,
            num_pre_layers=1,
            out_channels=self.d_model,
            is_v7=is_v7
        )

        if self.config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE:
            # The input is all zeros, so we can just use a single layer MLP.
            self.light_mlps = common_layers.build_mlps(
                c_in=constants.TRAFFIC_LIGHT_STATE_DIM, mlp_channels=[self.d_model], ret_before_act=True, is_v7=is_v7
            )
        else:
            self.light_mlps = common_layers.build_mlps(
                c_in=constants.TRAFFIC_LIGHT_STATE_DIM * SCENE_INPUT_TIME_STEPS,
                mlp_channels=[self.d_model] * 3,
                ret_before_act=True,
                is_v7=is_v7
            )

        simple_relation_factor = self.config.SIMPLE_RELATION_FACTOR
        simple_relation = self.config.SIMPLE_RELATION
        if self.config.SIMPLE_RELATION:
            relation_d_model = self.d_model // simple_relation_factor
            self.relation_embed = fourier_embedding.FourierEmbedding(
                input_dim=3, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
            )

            if self.config.MODEL.ADD_RELATION_TO_V:
                self.relation_embed_v = fourier_embedding.FourierEmbedding(
                    input_dim=3, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                )

        else:
            self.relation_embed = fourier_embedding.FourierEmbedding(
                input_dim=4, hidden_dim=self.d_model, num_freq_bands=64, is_v7=is_v7
            )
        assert self.config.MODEL.NAME in ['gpt']
        self.encoder = SelfAttTransformerEncoder(
            decoder_layer=SelfAttTransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                simple_relation=simple_relation,
                simple_relation_factor=simple_relation_factor,
                dropout=dropout,
                is_v7=is_v7,
                update_relation=self.config.UPDATE_RELATION,
                add_relation_to_v=self.config.MODEL.ADD_RELATION_TO_V,
                remove_rel_norm=self.config.REMOVE_REL_NORM,
            ),
            num_layers=self.num_layers,
        )
        self.out = common_layers.build_mlps(
            c_in=self.d_model, mlp_channels=[self.d_model], ret_before_act=True, is_v7=is_v7
        )
        self.out_prenorm = nn.LayerNorm(self.d_model)

        self.use_agent_history = not self.config.REMOVE_AGENT_FROM_SCENE_ENCODER
        if self.use_agent_history:
            self.agent_pe = common_layers.Tokenizer(
                num_actions=self.config.PREPROCESSING.MAX_AGENTS, d_model=self.d_model, add_one_more_action=False
            )
            self.agent_mlps = common_layers.build_mlps(
                c_in=constants.AGENT_STATE_DIM * SCENE_INPUT_TIME_STEPS,
                mlp_channels=[self.d_model] * 3,
                ret_before_act=True,
            )

    def encode_agent_history(self, input_dict):
        B, T, N, D_agent = input_dict["encoder/agent_feature"].shape
        in_evaluation = input_dict["in_evaluation"][0].item()

        # ===== Embed agent feature =====
        agent_feature = input_dict["encoder/agent_feature"]
        agent_valid_mask = input_dict["encoder/agent_valid_mask"]
        agent_position = input_dict["encoder/agent_position"]
        agent_heading = input_dict["encoder/agent_heading"]
        agent_id = input_dict["encoder/agent_id"].clone()
        assert agent_feature.shape[:3] == agent_position.shape[:3] == agent_valid_mask.shape[:3]
        agent_feature = (agent_feature[:, :self.history_steps] * agent_valid_mask[:, :self.history_steps, ..., None])
        agent_feature = collapse_time(agent_feature)
        agent_token = self.agent_mlps(agent_feature)  # (B, N, D)
        if in_evaluation:
            # Exempt filtering for maximum number of agents, so agent_id might be out of bound.
            agent_id = mode_agent_id(agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=True)
            # Exempt filtering for maximum number of agents, so agent_id might be out of bound.
            modeled_agent_id = mode_agent_id(
                input_dict["encoder/modeled_agent_id"].clone(),
                self.config.PREPROCESSING.MAX_AGENTS,
                fill_negative_1=True
            )
        else:
            modeled_agent_id = input_dict["encoder/modeled_agent_id"].clone()

        if self.config.MODEL.RANDOMIZE_AGENT_ID:
            weights = torch.ones(self.config.PREPROCESSING.MAX_AGENTS).expand(B, -1)
            if N > self.config.PREPROCESSING.MAX_AGENTS:
                new_encoder_agent_id = torch.full_like(agent_id, -1)
                num_samples = self.config.PREPROCESSING.MAX_AGENTS
                new_encoder_agent_id[:, :num_samples] = torch.multinomial(
                    weights, num_samples=num_samples, replacement=False
                ).to(agent_id)
                assert (agent_id[:, num_samples:] == -1).all()
            else:
                num_samples = N
                new_encoder_agent_id = torch.multinomial(
                    weights, num_samples=num_samples, replacement=False
                ).to(agent_id)
                new_encoder_agent_id[agent_id == -1] = -1
            input_dict["encoder/randomized_agent_id"] = new_encoder_agent_id
            agent_id = new_encoder_agent_id

            modeled_agent_mask = modeled_agent_id == -1
            modeled_agent_id[modeled_agent_mask] = N - 1  # Quick workaround
            new_modeled_agent_id = torch.gather(new_encoder_agent_id, dim=1, index=modeled_agent_id)
            new_modeled_agent_id[modeled_agent_mask] = -1
            input_dict["encoder/randomized_modeled_agent_id"] = new_modeled_agent_id
            modeled_agent_id = new_modeled_agent_id
        else:
            raise ValueError("Please turn on MODEL.RANDOMIZE_AGENT_ID=True")

        agent_pe = self.agent_pe(agent_id)  # (B, N, D)
        agent_token += agent_pe
        assert agent_token.shape == (B, N, self.d_model)

        agent_pos = find_last_valid(agent_position[:, :self.history_steps], agent_valid_mask[:, :self.history_steps])[:,
                                                                                                                      0]
        agent_mask = agent_valid_mask[:, :self.history_steps].any(dim=1)
        agent_heading = find_last_valid(
            agent_heading[:, :self.history_steps, ..., None], agent_valid_mask[:, :self.history_steps]
        )[:, 0, :, 0]

        input_dict["encoder/modeled_agent_pe"] = self.agent_pe(modeled_agent_id)

        return agent_token, agent_pos, agent_mask, agent_heading

    def forward(self, input_dict):
        # ===== Get shape =====
        B, M, num_vector, D_vector = input_dict["encoder/map_feature"].shape
        L, D_light = input_dict["encoder/traffic_light_feature"].shape[-2:]

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
            if self.config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE:
                traffic_light_feature = traffic_light_feature * traffic_light_valid_mask[..., None]
                traffic_light_token = self.light_mlps(traffic_light_feature)
            else:
                traffic_light_feature = (
                    traffic_light_feature[:, :self.history_steps] *
                    traffic_light_valid_mask[:, :self.history_steps, ..., None]
                )
                traffic_light_feature = collapse_time(traffic_light_feature)
                traffic_light_token = self.light_mlps(traffic_light_feature)
        else:
            traffic_light_token = traffic_light_feature.new_zeros([B, L, self.d_model])
        assert traffic_light_token.shape == (B, L, self.d_model), (traffic_light_token.shape, B, L, self.d_model)
        if self.config.PREPROCESSING.REMOVE_TRAFFIC_LIGHT_STATE:
            assert traffic_light_valid_mask.ndim == 2
            tlmask = traffic_light_valid_mask
        else:
            tlmask = traffic_light_valid_mask[:, :self.history_steps].any(dim=1)

        x = [map_token, traffic_light_token]
        x_pos = [map_position, traffic_light_position]
        x_heading = [map_heading, traffic_light_heading]
        x_mask = [map_token_valid_mask, tlmask]
        if self.use_agent_history:
            agent_token, agent_pos, agent_mask, agent_heading = self.encode_agent_history(input_dict=input_dict)
            x.append(agent_token)
            x_pos.append(agent_pos)
            x_mask.append(agent_mask)
            x_heading.append(agent_heading)

        # ===== Call transformer layers =====
        x = torch.concatenate(x, dim=1)
        x_pos = torch.concatenate(x_pos, dim=1)
        x_heading = torch.concatenate(x_heading, dim=1)
        x_mask = torch.concatenate(x_mask, dim=1)

        # There something wrong in waymo test set:
        # https://github.com/waymo-research/waymo-open-dataset/issues/772
        # And the line below might cause issue if we don't skip scenario before entering here.
        assert torch.all(x_mask.sum(dim=-1) > 0)

        if self.config.SIMPLE_RELATION:
            relation_func = relation.compute_relation_simple_relation
        else:
            relation_func = relation.compute_relation
        rel_feat, rel_mask, indices = relation_func(
            query_pos=x_pos,
            query_heading=x_heading,
            query_valid_mask=x_mask,
            key_pos=x_pos,
            key_heading=x_heading,
            key_valid_mask=x_mask,
            hidden_dim=self.d_model,
            causal_valid_mask=None,
            knn=self.config.MODEL.KNN,
            max_distance=self.config.MODEL.S2S_DISTANCE,
            gather=False,
            return_pe=False,
            non_agent_relation=True,
            per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION,
        )
        rel_pe = utils.unwrap(self.relation_embed(rel_feat[rel_mask]), rel_mask)
        rel_pe_v = None
        if self.config.MODEL.ADD_RELATION_TO_V:
            rel_pe_v = utils.unwrap(self.relation_embed_v(rel_feat[rel_mask]), rel_mask)
        scene_info = get_edge_info(attn_valid_mask=rel_mask, rel_pe_cross=rel_pe, rel_pe_cross_v=rel_pe_v)

        # print("rel_mask.shape", rel_mask.shape, rel_mask.sum(-1).float().mean(),  rel_mask.float().mean())

        #
        # from torch.nn.attention.flex_attention import (
        #     _DEFAULT_SPARSE_BLOCK_SIZE,
        #     create_block_mask,
        #     create_mask,
        #     flex_attention,
        # )
        # from triton.testing import do_bench

        # torch.set_default_device("cuda")
        # torch.manual_seed(0)
        #
        # torch._dynamo.config.cache_size_limit = 1000
        #
        # Compile the flex_attention function
        # flex_attention = torch.compile(flex_attention, dynamic=False)

        # Define `score_mod` without precomputing softmax_bias

        # Q = x.reshape(B, x.shape[1], 8, -1).swapaxes(1, 2)
        # Relation = rel_pe.reshape(B, rel_pe.shape[1], rel_pe.shape[2], 8, -1)
        # def score_mod(score, b, h, q_idx, kv_idx):
        #     bias = Q[b, h, q_idx] @ Relation[b, q_idx, kv_idx, h]
        #     return score + bias
        #
        #
        #
        # flex_attention(
        #     query=Q,
        #     key=x.reshape(B, x.shape[1], 8, -1).swapaxes(1, 2),
        #     value=x.reshape(B, x.shape[1], 8, -1).swapaxes(1, 2),
        #     score_mod=score_mod,
        # )

        x = self.encoder(
            scene_tokens=x,
            scene_info=scene_info,
            edge_features=scene_info["edge_features"],
            edge_features_v=scene_info["edge_features_v"]
        )
        x = self.out_prenorm(x[x_mask])
        x = self.out(x)  # .reshape(list(x.shape[:-1]) + [self.d_model])
        x = utils.unwrap(x, x_mask)
        input_dict["encoder/scenario_token"] = x
        input_dict["encoder/map_token"] = x[:, :M]
        input_dict["encoder/scenario_position"] = x_pos
        input_dict["encoder/scenario_heading"] = x_heading
        input_dict["encoder/scenario_valid_mask"] = x_mask
        return input_dict
