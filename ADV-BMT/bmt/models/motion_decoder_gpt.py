import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from bmt.dataset import constants
from bmt.dataset.preprocess_action_label import SafetyAction
from bmt.models import relation
from bmt.models.layers import common_layers, fourier_embedding
from bmt.models.layers.gpt_decoder_layer import MultiCrossAttTransformerDecoderLayer, MultiCrossAttTransformerDecoder
from bmt.models.motion_decoder import create_causal_mask
from bmt.models.scene_encoder import mode_agent_id
from bmt.tokenization import get_action_dim, get_tokenizer, START_ACTION, END_ACTION
from bmt.utils import utils


def get_edge_info(attn_valid_mask, rel_pe_cross, rel_pe_cross_v=None):
    B, Lq, Lk = attn_valid_mask.shape
    edge_index, _ = dense_to_sparse(attn_valid_mask.swapaxes(1, 2).contiguous())
    assert edge_index.numel() > 0, (edge_index.shape, attn_valid_mask.sum())
    assert edge_index[0].max() < B * Lk, f"{edge_index[0].max()} >= {B * Lk}"
    assert edge_index[1].max() < B * Lq, f"{edge_index[1].max()} >= {B * Lq}"

    if rel_pe_cross is not None:
        batch_ind = edge_index[1] // Lq
        q_ind = edge_index[1] % Lq
        batch_ind_k = edge_index[0] // Lk
        k_ind = edge_index[0] % Lk
        assert torch.all(batch_ind == batch_ind_k)
        edge_features = rel_pe_cross[batch_ind, q_ind, k_ind]
    else:
        edge_features = None

    if rel_pe_cross_v is not None:
        assert rel_pe_cross is not None
        edge_features_v = rel_pe_cross_v[batch_ind, q_ind, k_ind]
    else:
        edge_features_v = None

    return {
        "edge_index": edge_index,
        "edge_features": edge_features,
        "edge_features_v": edge_features_v,
        # "attn_valid_mask": attn_valid_mask,
        # "relation": rel_pe_cross,
        # "relation_v": rel_pe_cross_v,
    }  # "relation": rel_pe_cross, "attn_valid_mask": attn_valid_mask}


class MotionDecoderGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS
        self.num_actions = get_action_dim(self.config)
        dropout = self.config.MODEL.DROPOUT
        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        # use_condition = self.config.ACTION_LABEL.USE_ACTION_LABEL or self.config.ACTION_LABEL.USE_SAFETY_LABEL
        # self.use_condition = use_condition
        assert self.config.MODEL.NAME in ['gpt']
        self.add_pe_for_token = self.config.MODEL.get('ADD_PE_FOR_TOKEN', False)
        assert self.add_pe_for_token
        use_adaln = self.config.USE_ADALN
        self.use_adaln = use_adaln

        simple_relation = self.config.SIMPLE_RELATION
        simple_relation_factor = self.config.SIMPLE_RELATION_FACTOR
        is_v7 = self.config.MODEL.IS_V7
        self.is_v7 = is_v7
        self.decoder = MultiCrossAttTransformerDecoder(
            decoder_layer=MultiCrossAttTransformerDecoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dropout=dropout,
                use_adaln=use_adaln,
                simple_relation=simple_relation,
                simple_relation_factor=simple_relation_factor,
                is_v7=is_v7,
                update_relation=self.config.UPDATE_RELATION,
                add_relation_to_v=self.config.MODEL.ADD_RELATION_TO_V,
                remove_rel_norm=self.config.REMOVE_REL_NORM
            ),
            num_layers=num_decoder_layers,
            d_model=d_model,
        )
        self.prediction_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, self.num_actions], ret_before_act=True, is_v7=is_v7, zero_init=is_v7
        )
        if self.use_adaln:
            self.prediction_adaln_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))
        else:
            self.prediction_prenorm = nn.LayerNorm(d_model)

        # if self.config.BACKWARD_PREDICTION:
        # if is_v7:
        #     raise ValueError()
        # self.prediction_backward_head = common_layers.build_mlps(
        #     c_in=d_model, mlp_channels=[d_model, d_model, self.num_actions], ret_before_act=True
        # )
        # self.prediction_backward_prenorm = nn.LayerNorm(d_model)

        if self.config.ADD_CONTOUR_RELATION:

            if self.config.SIMPLE_RELATION:
                relation_d_model = d_model // simple_relation_factor

                self.relation_embed_a2a = fourier_embedding.FourierEmbedding(
                    input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                )
                self.relation_embed_a2t = fourier_embedding.FourierEmbedding(
                    input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                )
                self.relation_embed_a2s = fourier_embedding.FourierEmbedding(
                    input_dim=3, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                )

                if self.config.MODEL.ADD_RELATION_TO_V:
                    self.relation_embed_a2a_v = fourier_embedding.FourierEmbedding(
                        input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                    )
                    self.relation_embed_a2t_v = fourier_embedding.FourierEmbedding(
                        input_dim=12, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                    )
                    self.relation_embed_a2s_v = fourier_embedding.FourierEmbedding(
                        input_dim=3, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                    )

            else:
                relation_d_model = d_model

                self.relation_embed_a2a = fourier_embedding.FourierEmbedding(
                    input_dim=13, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                )
                self.relation_embed_a2t = fourier_embedding.FourierEmbedding(
                    input_dim=13, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                )
                self.relation_embed_a2s = fourier_embedding.FourierEmbedding(
                    input_dim=13, hidden_dim=relation_d_model, num_freq_bands=64, is_v7=is_v7
                )
        else:
            assert self.config.SIMPLE_RELATION is False
            self.relation_embed_a2a = fourier_embedding.FourierEmbedding(
                input_dim=5, hidden_dim=d_model, num_freq_bands=64, is_v7=is_v7
            )
            self.relation_embed_a2t = fourier_embedding.FourierEmbedding(
                input_dim=5, hidden_dim=d_model, num_freq_bands=64, is_v7=is_v7
            )
            self.relation_embed_a2s = fourier_embedding.FourierEmbedding(
                input_dim=5, hidden_dim=d_model, num_freq_bands=64, is_v7=is_v7
            )

        self.type_embed = common_layers.Tokenizer(
            num_actions=constants.NUM_TYPES, d_model=d_model, add_one_more_action=False
        )
        self.action_embed = common_layers.Tokenizer(
            num_actions=self.num_actions, d_model=d_model, add_one_more_action=True
        )
        self.shape_embed = common_layers.build_mlps(
            c_in=3, mlp_channels=[d_model, d_model], ret_before_act=True, is_v7=is_v7
        )

        if self.config.REMOVE_AGENT_FROM_SCENE_ENCODER:
            self.agent_id_embed = common_layers.Tokenizer(
                num_actions=self.config.PREPROCESSING.MAX_AGENTS, d_model=self.d_model, add_one_more_action=False
            )

        self.motion_embed = fourier_embedding.FourierEmbedding(
            input_dim=6, hidden_dim=d_model, num_freq_bands=64, is_v7=is_v7
        )

        tokenizer = get_tokenizer(self.config)
        motion_features = tokenizer.get_motion_feature()
        if tokenizer.use_type_specific_bins:
            motion_features = torch.cat([motion_features, torch.zeros(1, 3, 4)], dim=0)
        else:
            motion_features = torch.cat([motion_features, torch.zeros(1, 4)], dim=0)
        self.tokenizer = tokenizer
        self.register_buffer("motion_features", motion_features)

        # is start token? is end token (if any)? is padding token? is masked token?
        self.special_token_embed = common_layers.Tokenizer(
            num_actions=4, d_model=self.d_model, add_one_more_action=False
        )

        if self.config.BACKWARD_PREDICTION:
            self.in_backward_prediction_embed = common_layers.Tokenizer(
                num_actions=2, d_model=self.d_model, add_one_more_action=False
            )

        # self.use_action_label = config.ACTION_LABEL.USE_ACTION_LABEL or config.ACTION_LABEL.USE_SAFETY_LABEL
        if config.ACTION_LABEL.USE_ACTION_LABEL:
            raise ValueError("Not implemented")
            # self.action_label_tokenizer_turn = common_layers.Tokenizer(
            #     num_actions=TurnAction.num_actions, d_model=d_model, add_one_more_action=True
            # )
            # self.action_label_tokenizer_accel = common_layers.Tokenizer(
            #     num_actions=AccelerationAction.num_actions, d_model=d_model, add_one_more_action=True
            # )
        if config.ACTION_LABEL.USE_SAFETY_LABEL:
            self.action_label_tokenizer_safety = common_layers.Tokenizer(
                num_actions=SafetyAction.num_actions, d_model=d_model, add_one_more_action=True
            )
        if self.use_adaln:
            self.initialize_weights_for_adaln()

        # if self.is_v7:
        #     self.prediction_head[-1].weight.data.fill_(0)

    def initialize_weights_for_adaln(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.layers:
            nn.init.constant_(block.adaln_modulation[-1].weight, 0)
            nn.init.constant_(block.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.adaln_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_modulation[-1].bias, 0)

    def randomize_modeled_agent_id(self, data_dict, clip_agent_id=False):
        modeled_agent_id = data_dict["decoder/agent_id"]
        # batch_index = data_dict.get("batch_idx", None)
        if not self.config.MODEL.RANDOMIZE_AGENT_ID:
            if clip_agent_id:
                modeled_agent_id = mode_agent_id(
                    modeled_agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=True
                )
            return modeled_agent_id

        # assert batch_index is not None, "Need batch index to randomize agent id!"
        # batch_to_unique = {}
        # for i, b in enumerate(batch_index):
        #     b = b.item()
        #     if b not in batch_to_unique:
        #         batch_to_unique[b] = len(batch_to_unique)

        if clip_agent_id:
            modeled_agent_id = mode_agent_id(
                modeled_agent_id, self.config.PREPROCESSING.MAX_AGENTS, fill_negative_1=True
            )
        B, N = modeled_agent_id.shape
        weights = torch.ones(self.config.PREPROCESSING.MAX_AGENTS).expand(B, -1)
        if N > self.config.PREPROCESSING.MAX_AGENTS:
            num_samples = self.config.PREPROCESSING.MAX_AGENTS
            new_modeled_agent_id = torch.full_like(modeled_agent_id, num_samples - 1)
            new_modeled_agent_id[:, :num_samples] = torch.multinomial(
                weights, num_samples=num_samples, replacement=False
            ).to(modeled_agent_id)
            new_modeled_agent_id[modeled_agent_id == -1] = -1
        else:
            num_samples = N
            new_modeled_agent_id = torch.multinomial(
                weights, num_samples=num_samples, replacement=False
            ).to(modeled_agent_id)
            new_modeled_agent_id[modeled_agent_id == -1] = -1

        # Allocate same agent id to the same batch
        # return_modeled_agent_id = torch.full_like(modeled_agent_id, -1)
        # for i, b in enumerate(batch_index):
        #     b = b.item()
        #     return_modeled_agent_id[i] = new_modeled_agent_id[batch_to_unique[b]]
        # return return_modeled_agent_id
        return new_modeled_agent_id

    def forward(self, input_dict, use_cache=False, a2a_knn=None, a2t_knn=None, a2s_knn=None):
        in_evaluation = input_dict["in_evaluation"][0].item()

        # num_heads = self.num_heads
        # === Process scene embedding ===
        scene_token = input_dict["encoder/scenario_token"]
        scenario_valid_mask = input_dict["encoder/scenario_valid_mask"]

        # === Process action embedding ===
        input_action = input_dict["decoder/input_action"]
        modeled_agent_delta = input_dict["decoder/modeled_agent_delta"]
        B, T_skipped, N = input_action.shape

        if self.config.REMOVE_AGENT_FROM_SCENE_ENCODER:
            if in_evaluation:
                assert "decoder/randomized_modeled_agent_id" in input_dict, "Need to provide randomized modeled agent id for evaluation! Please call randomize_modeled_agent_id()"
                new_modeled_agent_id = input_dict["decoder/randomized_modeled_agent_id"]
            else:
                new_modeled_agent_id = self.randomize_modeled_agent_id(input_dict, clip_agent_id=False)
            modeled_agent_pe = self.agent_id_embed(new_modeled_agent_id)

            # print("modeled_agent_pe", new_modeled_agent_id[0])
        else:
            modeled_agent_pe = input_dict["encoder/modeled_agent_pe"]

        assert modeled_agent_pe.shape == (B, N, self.d_model), modeled_agent_pe.shape
        modeled_agent_pe = modeled_agent_pe[:, None].expand(B, T_skipped, N, self.d_model)

        action_valid_mask = input_dict["decoder/input_action_valid_mask"]
        assert action_valid_mask.shape == (B, T_skipped, N), (action_valid_mask.shape, (B, T_skipped, N))
        agent_pos = input_dict["decoder/modeled_agent_position"]
        agent_heading = input_dict["decoder/modeled_agent_heading"]
        # agent_vel = input_dict["decoder/modeled_agent_velocity"]

        # ===== Prepare input tokens =====
        if "decoder/input_step" not in input_dict:
            input_dict["decoder/input_step"] = torch.arange(T_skipped).to(input_action.device)
        agent_step = input_dict["decoder/input_step"].reshape(1, T_skipped, 1).expand(B, T_skipped, N)

        # Shape embedding and type embedding
        type_emb = self.type_embed(input_dict["decoder/agent_type"])[:, None].expand(B, T_skipped, N, self.d_model)
        shape_emb = self.shape_embed(input_dict["decoder/current_agent_shape"]
                                     )[:, None].expand(B, T_skipped, N, self.d_model)

        valid_actions = input_action[action_valid_mask]
        is_start_actions = valid_actions == START_ACTION
        special_tok = torch.full_like(valid_actions, 0).int()
        special_tok[is_start_actions] = 1
        valid_actions[is_start_actions] = -1
        if self.config.BACKWARD_PREDICTION:
            is_end_actions = valid_actions == END_ACTION
            special_tok[is_end_actions] = 2
            valid_actions[is_end_actions] = -1
        special_tok_emb = self.special_token_embed(special_tok)
        if self.config.BACKWARD_PREDICTION:
            if "in_backward_prediction" not in input_dict:
                input_dict["in_backward_prediction"] = valid_actions.new_zeros(B, T_skipped, N)
            in_backward_full = input_dict["in_backward_prediction"].reshape(B, 1, 1).expand(B, T_skipped, N)
            in_backward = in_backward_full[action_valid_mask]
            in_backward = in_backward.int()
            in_backward_prediction_embed = self.in_backward_prediction_embed(in_backward)
            special_tok_emb += in_backward_prediction_embed
        action_emb = self.action_embed(valid_actions)

        # agent_type = agent_type[:, None].expand(B, T_skipped, N)
        if self.tokenizer.use_type_specific_bins:
            agent_type = input_dict["decoder/agent_type"]
            agent_type = agent_type - 1
            agent_type[agent_type < 0] = 0
            agent_type = agent_type.reshape(B, 1, N).expand(B, T_skipped, N)
            agent_type = agent_type[action_valid_mask]  # Already flattened
            agent_type = agent_type.reshape(-1, 1, 1, 1).expand(-1, self.motion_features.shape[0], 1, 4)
            motion_feat = self.motion_features.reshape(1, -1, 3, 4).expand(agent_type.shape[0], -1, 3, 4)
            motion_feat = torch.gather(motion_feat, dim=-2, index=agent_type).squeeze(-2)
        else:
            motion_feat = self.motion_features.reshape(1, -1, 4).expand(valid_actions.shape[0], -1, 4)
        valid_actions[valid_actions < 0] = self.num_actions
        valid_actions = valid_actions.reshape(-1, 1, 1).expand(-1, 1, 4)
        motion_feat = torch.gather(motion_feat, dim=-2, index=valid_actions).squeeze(-2)

        motion_feat = torch.cat([motion_feat, modeled_agent_delta[action_valid_mask]], dim=-1)

        action_token = self.motion_embed(
            continuous_inputs=motion_feat,
            categorical_embs=[
                special_tok_emb, modeled_agent_pe[action_valid_mask], type_emb[action_valid_mask],
                shape_emb[action_valid_mask], action_emb
            ]
        )
        action_token = utils.unwrap(action_token, action_valid_mask)
        assert action_token.shape == (B, T_skipped, N, self.d_model)
        assert action_valid_mask.shape == (B, T_skipped, N)

        # ===== Get agent-condition relation =====
        condition_token = None
        if self.config.ACTION_LABEL.USE_SAFETY_LABEL:
            action_label_safety = self.action_label_tokenizer_safety(input_dict["decoder/label_safety"])
            condition_token = action_label_safety[:, None]
            if self.use_adaln:
                pass
            else:
                action_token += condition_token

        # ===== Get agent-temporal relation =====
        # BTND -> BNTD
        agent_pos_bntd = torch.permute(agent_pos, [0, 2, 1, 3])
        agent_heading_bnt = torch.permute(agent_heading, [0, 2, 1])
        agent_mask_bnt = torch.permute(action_valid_mask, [0, 2, 1])
        agent_step_bnt = torch.permute(agent_step, [0, 2, 1])
        # agent_vel_bnt = torch.permute(agent_vel, [0, 2, 1, 3])
        if use_cache:
            self.update_cache(input_dict)

            agent_pos_with_history = input_dict["decoder/modeled_agent_position_history"]
            agent_heading_with_history = input_dict["decoder/modeled_agent_heading_history"]
            agent_mask_with_history = input_dict["decoder/modeled_agent_valid_mask_history"]
            agent_step_with_history = input_dict["decoder/modeled_agent_step_history"]
            agent_vel_with_history = input_dict["decoder/modeled_agent_velocity_history"]
            real_T = agent_mask_with_history.shape[1]
            key_pos = torch.permute(agent_pos_with_history, [0, 2, 1, 3]).flatten(0, 1)
            # key_vel = torch.permute(agent_vel_with_history, [0, 2, 1, 3]).flatten(0, 1)
            key_heading = torch.permute(agent_heading_with_history, [0, 2, 1]).flatten(0, 1)
            key_mask = torch.permute(agent_mask_with_history, [0, 2, 1]).flatten(0, 1)
            causal_valid_mask = None
            key_step = agent_step_with_history.reshape(1, 1, -1).expand(B, N, -1).flatten(0, 1)
        else:
            real_T = T_skipped
            # key_vel = agent_vel_bnt.flatten(0, 1)
            key_pos = agent_pos_bntd.flatten(0, 1)
            key_heading = agent_heading_bnt.flatten(0, 1)
            key_mask = agent_mask_bnt.flatten(0, 1)
            key_step = agent_step_bnt.flatten(0, 1)
            causal_valid_mask = create_causal_mask(T=real_T, N=1, is_valid_mask=True).to(action_token.device)

        assert agent_pos_bntd.shape == (B, N, T_skipped, 2)

        a2t_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            agent_shape_no_time = input_dict["decoder/current_agent_shape"
                                             ]  #.reshape(B, 1, N, 3).expand(B, real_T, N, 3)
            agent_length = agent_shape_no_time[..., 0]
            agent_width = agent_shape_no_time[..., 1]
            a2t_kwargs = dict(
                include_contour=True,
                query_width=agent_width.flatten(0, 1).unsqueeze(1).expand(-1, T_skipped),
                query_length=agent_length.flatten(0, 1).unsqueeze(1).expand(-1, T_skipped),
                key_width=agent_width.flatten(0, 1).unsqueeze(1).expand(-1, real_T),
                key_length=agent_length.flatten(0, 1).unsqueeze(1).expand(-1, real_T),
                non_agent_relation=False,
                per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION
            )

        if self.config.SIMPLE_RELATION:
            relation_func = relation.compute_relation_simple_relation
        else:
            relation_func = relation.compute_relation

        a2t_rel_feat, a2t_mask, _ = relation_func(
            query_pos=agent_pos_bntd.flatten(0, 1),  # BN, T, D
            query_heading=agent_heading_bnt.flatten(0, 1),
            query_valid_mask=agent_mask_bnt.flatten(0, 1),
            query_step=agent_step_bnt.flatten(0, 1),
            key_pos=key_pos,  # BN, T_full, D
            key_heading=key_heading,
            key_valid_mask=key_mask,
            key_step=key_step,
            hidden_dim=self.d_model,
            causal_valid_mask=causal_valid_mask,
            knn=None,
            max_distance=None,
            return_pe=False,
            # key_vel=key_vel,
            # query_vel=agent_vel_bnt.flatten(0, 1),
            **a2t_kwargs
        )
        a2t_rel_pe = utils.unwrap(self.relation_embed_a2t(a2t_rel_feat[a2t_mask]), a2t_mask)
        a2t_rel_pe_v = None
        if self.config.MODEL.ADD_RELATION_TO_V:
            a2t_rel_pe_v = utils.unwrap(self.relation_embed_a2t_v(a2t_rel_feat[a2t_mask]), a2t_mask)
        a2t_info = get_edge_info(attn_valid_mask=a2t_mask, rel_pe_cross=a2t_rel_pe, rel_pe_cross_v=a2t_rel_pe_v)

        # print("===")
        # print("a2t_mask.shape", a2t_mask.shape, a2t_mask.sum(-1).float().mean(), a2t_mask.float().mean())

        # ===== Get agent-agent relation =====
        a2a_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            w = agent_width.unsqueeze(1).expand(B, T_skipped, N).flatten(0, 1)
            l = agent_length.unsqueeze(1).expand(B, T_skipped, N).flatten(0, 1)
            a2a_kwargs = dict(
                include_contour=True,
                query_width=w,
                query_length=l,
                key_width=w,
                key_length=l,
                non_agent_relation=False,
                per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION
            )
        a2a_rel_feat, a2a_mask, _ = relation_func(
            query_pos=agent_pos.flatten(0, 1),  # BT, N, D
            query_heading=agent_heading.flatten(0, 1),
            query_valid_mask=action_valid_mask.flatten(0, 1),
            query_step=agent_step.flatten(0, 1),
            key_pos=agent_pos.flatten(0, 1),
            key_heading=agent_heading.flatten(0, 1),
            key_valid_mask=action_valid_mask.flatten(0, 1),
            key_step=agent_step.flatten(0, 1),
            hidden_dim=self.d_model,
            causal_valid_mask=None,
            knn=a2a_knn if a2a_knn is not None else self.config.MODEL.A2A_KNN,
            max_distance=self.config.MODEL.A2A_DISTANCE,
            return_pe=False,
            # query_vel=agent_vel.flatten(0, 1),
            # key_vel=agent_vel.flatten(0, 1),
            **a2a_kwargs
        )
        a2a_rel_pe = utils.unwrap(self.relation_embed_a2a(a2a_rel_feat[a2a_mask]), a2a_mask)
        a2a_rel_pe_v = None
        if self.config.MODEL.ADD_RELATION_TO_V:
            a2a_rel_pe_v = utils.unwrap(self.relation_embed_a2a_v(a2a_rel_feat[a2a_mask]), a2a_mask)
        a2a_info = get_edge_info(attn_valid_mask=a2a_mask, rel_pe_cross=a2a_rel_pe, rel_pe_cross_v=a2a_rel_pe_v)

        # print("a2a_mask.shape", a2a_mask.shape, a2a_mask.sum(-1).float().mean(),  a2a_mask.float().mean())

        # ===== Get agent-scene relation =====
        a2s_kwargs = {}
        if self.config.ADD_CONTOUR_RELATION:
            w = agent_width.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            l = agent_length.unsqueeze(1).expand(B, T_skipped, N).flatten(1, 2)
            kw = torch.zeros_like(input_dict["encoder/scenario_position"][..., 0])
            a2s_kwargs = dict(
                include_contour=True,
                query_width=w,
                query_length=l,
                key_width=kw,
                key_length=kw,
                non_agent_relation=True,
                per_contour_point_relation=self.config.MODEL.PER_CONTOUR_POINT_RELATION
            )
        a2s_rel_feat, a2s_mask, a2s_indices = relation_func(
            query_pos=agent_pos.flatten(1, 2),  # B, TN, D
            query_heading=agent_heading.flatten(1, 2),
            query_valid_mask=action_valid_mask.flatten(1, 2),
            query_step=agent_step.flatten(1, 2),
            key_pos=input_dict["encoder/scenario_position"],  # [..., :2],
            key_heading=input_dict["encoder/scenario_heading"],
            key_valid_mask=scenario_valid_mask,
            key_step=agent_pos.new_zeros(B, input_dict["encoder/scenario_position"].shape[1]),
            hidden_dim=self.d_model,
            causal_valid_mask=None,
            knn=a2s_knn if a2s_knn is not None else self.config.MODEL.A2S_KNN,
            max_distance=self.config.MODEL.A2S_DISTANCE,
            gather=False,
            return_pe=False,
            **a2s_kwargs
        )
        a2s_rel_pe = utils.unwrap(self.relation_embed_a2s(a2s_rel_feat[a2s_mask]), a2s_mask)
        a2s_rel_pe_v = None
        if self.config.MODEL.ADD_RELATION_TO_V:
            a2s_rel_pe_v = utils.unwrap(self.relation_embed_a2s_v(a2s_rel_feat[a2s_mask]), a2s_mask)
        a2s_info = get_edge_info(attn_valid_mask=a2s_mask, rel_pe_cross=a2s_rel_pe, rel_pe_cross_v=a2s_rel_pe_v)

        # print("a2s_mask.shape", a2s_mask.shape, a2s_mask.sum(-1).float().mean(),  a2s_mask.float().mean())

        # === Call models ===
        past_key_value_list = None
        if use_cache:
            # Cache from last rollout
            if "decoder/cache" in input_dict:
                past_key_value_list = input_dict["decoder/cache"]

        decoded_tokens = self.decoder(
            agent_token=action_token,
            scene_token=scene_token,
            a2a_info=a2a_info,
            a2t_info=a2t_info,
            a2s_info=a2s_info,
            condition_token=condition_token if self.use_adaln else None,
            use_cache=use_cache,  # We don't need decoder to take care cache.
            past_key_value_list=past_key_value_list
        )

        if use_cache:
            decoded_tokens, past_key_value_list = decoded_tokens
            for l in past_key_value_list:
                if l:
                    l.append((B * N, real_T))
            input_dict["decoder/cache"] = past_key_value_list

        if self.use_adaln:
            output_tokens = self.prediction_adaln_norm(decoded_tokens[action_valid_mask])
            shift, scale = self.adaln_modulation(output_tokens).chunk(2, dim=-1)
            output_tokens = utils.modulate(output_tokens, shift, scale)
        else:
            output_tokens = self.prediction_prenorm(decoded_tokens[action_valid_mask])
        logits = utils.unwrap(self.prediction_head(output_tokens), action_valid_mask)

        # if self.config.BACKWARD_PREDICTION:
        #     output_tokens_backward = self.prediction_backward_prenorm(decoded_tokens[action_valid_mask])
        #     logits_backward = utils.unwrap(self.prediction_backward_head(output_tokens_backward), action_valid_mask)
        #
        #     logits = torch.where(
        #         in_backward_full.unsqueeze(-1).expand(-1, -1, -1, logits_backward.shape[-1]), logits_backward, logits
        #     )

        # if self.is_v7:
        #     logits = 30 * torch.tanh(logits / 30)

        assert logits.shape == (B, T_skipped, N, self.num_actions)
        input_dict["decoder/output_logit"] = logits

        # from torch.cuda import memory_snapshot
        #
        # snapshot = memory_snapshot()
        # # This will show a detailed report on allocations in JSON format
        # print(snapshot)

        return input_dict

    def update_cache(self, input_dict):
        assert self.config.EVALUATION.USE_CACHE
        if "decoder/modeled_agent_position_history" not in input_dict:
            input_dict["decoder/modeled_agent_position_history"] = input_dict["decoder/modeled_agent_position"].clone()
            input_dict["decoder/modeled_agent_velocity_history"] = input_dict["decoder/modeled_agent_velocity"].clone()
            input_dict["decoder/modeled_agent_heading_history"] = input_dict["decoder/modeled_agent_heading"].clone()
            input_dict["decoder/modeled_agent_valid_mask_history"] = input_dict["decoder/input_action_valid_mask"
                                                                                ].clone()
            input_dict["decoder/modeled_agent_step_history"] = input_dict["decoder/input_step"].clone()
        else:
            input_dict["decoder/modeled_agent_position_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_position_history"], input_dict["decoder/modeled_agent_position"]],
                dim=1
            )
            input_dict["decoder/modeled_agent_velocity_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_velocity_history"], input_dict["decoder/modeled_agent_velocity"]],
                dim=1
            )
            input_dict["decoder/modeled_agent_heading_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_heading_history"], input_dict["decoder/modeled_agent_heading"]],
                dim=1
            )
            input_dict["decoder/modeled_agent_valid_mask_history"] = torch.cat(
                [
                    input_dict["decoder/modeled_agent_valid_mask_history"],
                    input_dict["decoder/input_action_valid_mask"],
                ],
                dim=1
            )
            input_dict["decoder/modeled_agent_step_history"] = torch.cat(
                [input_dict["decoder/modeled_agent_step_history"], input_dict["decoder/input_step"]], dim=0
            )
