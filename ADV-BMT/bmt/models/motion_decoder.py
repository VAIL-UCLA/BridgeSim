import torch
import torch.nn as nn

from bmt.dataset.preprocess_action_label import TurnAction, AccelerationAction, SafetyAction
from bmt.models.layers import common_layers, position_encoding_utils
from bmt.models.layers.decoder_layer import TransformerDecoder, TransformerDecoderLayer
from bmt.tokenization import get_action_dim, get_tokenizer, START_ACTION
from bmt.utils import unwrap


def create_causal_mask(T, N, is_valid_mask=False):
    """ Create the causal mask for a flattened token sequence. Tokens will not attend to future ids. Tokens for the
    agents in the same step can attend to each other.

    row: a query
    col: a key

    So for mask[100] it should see more keys than mask[0].

    Note that all +1 positions will be filled -inf.

    Args:
        T: Number of steps
        N: Number of agents (padded to fit different batches)

    Returns:
        Causal mask in shape: (T*N, T*N), wherein 1s represent the ids to be ignored.
    """
    block = torch.ones(N, N, dtype=torch.bool)
    causal_mask = torch.kron(torch.tril(torch.ones(T, T, dtype=torch.bool)), block)
    if is_valid_mask:
        return causal_mask
    else:
        return ~causal_mask


class MotionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = d_model = self.config.MODEL.D_MODEL
        num_decoder_layers = self.config.MODEL.NUM_DECODER_LAYERS
        self.add_pe_for_static_features = self.config.MODEL.get('ADD_PE_FOR_STATIC_FEATURE', False)
        assert self.add_pe_for_static_features is False
        self.num_actions = get_action_dim(self.config)

        num_pred_steps = 16 + 1

        dropout = self.config.MODEL['DROPOUT_OF_ATTN']

        pre_projection = self.config.MODEL['PRE_PROJECTION']

        # TODO: better name
        self.relative_pe = self.config.MODEL['RELATIVE_PE_DECODER']

        self.num_heads = self.config.MODEL.NUM_ATTN_HEAD
        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=d_model,
                nhead=self.num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="relu",
                pre_projection=pre_projection
            ),
            num_layers=num_decoder_layers,
            relative_pe=self.relative_pe,
            d_model=d_model,
            self_attention_knn=self.config.MODEL['SELF_ATTN_KNN'],
            cross_attention_knn=self.config.MODEL['CROSS_ATTN_KNN'],
        )
        self.prediction_head = common_layers.build_mlps(
            c_in=d_model, mlp_channels=[d_model, d_model, self.num_actions], ret_before_act=True
        )

        self.step_pe = nn.Embedding(num_pred_steps, d_model)

        self.add_pe_for_token = self.config.MODEL.get('ADD_PE_FOR_TOKEN', False)
        self.tokenizer = common_layers.Tokenizer(num_actions=self.num_actions, d_model=d_model)
        if self.add_pe_for_token:
            tokenizer = get_tokenizer(self.config)
            # pe.shape = (num_actions, d_model)
            pe = position_encoding_utils.gen_sineembed_for_position(
                torch.from_numpy(tokenizer.bin_centers_flat)[None], hidden_dim=d_model
            )[0].float()

            self.tokenizer.tokens.weight = nn.Parameter(torch.cat([pe, self.tokenizer.tokens.weight[-1:]]))
            self.tokenizer.tokens.requires_grad_(False)

            self.tokenizer_mlp = common_layers.build_mlps(
                c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True
            )
        else:
            self.tokenizer_mlp = None

        self.use_action_label = config.ACTION_LABEL.USE_ACTION_LABEL
        if self.use_action_label:
            self.action_label_tokenizer_turn = common_layers.Tokenizer(
                num_actions=TurnAction.num_actions, d_model=d_model
            )
            self.action_label_tokenizer_accel = common_layers.Tokenizer(
                num_actions=AccelerationAction.num_actions, d_model=d_model
            )
        self.use_safety_label = config.ACTION_LABEL.USE_SAFETY_LABEL
        if self.use_safety_label:
            self.action_label_tokenizer_safety = common_layers.Tokenizer(
                num_actions=SafetyAction.num_actions, d_model=d_model
            )
        self.use_condition = self.use_safety_label or self.use_action_label

    def forward(self, input_dict, use_cache=False):
        # === Process scene embedding ===
        scene_token = input_dict["encoder/scenario_token"]
        scenario_valid_mask = input_dict["encoder/scenario_valid_mask"]
        modeled_agent_pe = input_dict["encoder/modeled_agent_pe"]
        scene_padding_mask = ~scenario_valid_mask

        # === Process action embedding ===
        input_action = input_dict["decoder/input_action"]
        B, T_skipped, N = input_action.shape
        action_valid_mask = input_dict["decoder/input_action_valid_mask"]
        assert action_valid_mask.shape == (B, T_skipped, N)

        input_action[input_action == START_ACTION] = -1

        action_token = self.tokenizer(input_action)  # (B, T_skipped, N, D)
        if self.add_pe_for_token:
            action_token = self.tokenizer_mlp(action_token)
        assert action_token.shape == (B, T_skipped, N, self.d_model)

        # Add PE to input action
        if "decoder/input_step" not in input_dict:
            input_dict["decoder/input_step"] = torch.arange(T_skipped).to(action_token.device)
        assert input_dict["decoder/input_step"].ndim == 1
        step_pe = self.step_pe(input_dict["decoder/input_step"])
        # print('input_dict["decoder/input_step"]', input_dict["decoder/input_step"])
        action_token += step_pe.reshape(1, T_skipped, 1, self.d_model)

        assert action_token.shape == (B, T_skipped, N, self.d_model)
        assert modeled_agent_pe.shape == (B, N, self.d_model), modeled_agent_pe.shape
        action_token += modeled_agent_pe[:, None]

        if self.add_pe_for_static_features:
            action_token += input_dict["encoder/modeled_agent_type_pe"][:, None]

        if self.use_action_label:
            action_label_turn = self.action_label_tokenizer_turn(input_dict["decoder/label_turning"])
            action_label_accel = self.action_label_tokenizer_accel(input_dict["decoder/label_acceleration"])
            action_token += action_label_turn[:, None]
            action_token += action_label_accel[:, None]

        if self.use_safety_label:
            action_label_safety = self.action_label_tokenizer_safety(input_dict["decoder/label_safety"])
            action_token += action_label_safety[:, None]

        action_casual_mask = create_causal_mask(
            T=T_skipped, N=N
        ).to(action_token.device)  # (B, T_skipped*N, T_skipped*N)

        # Just remove invalid actions
        action_token = action_token * action_valid_mask[..., None]

        action_padding_mask = ~action_valid_mask  # (T_skipped, N)
        # Flatten action token from (B, T_skipped, N, D) to (B, T_skipped*N, D)
        action_token = action_token.flatten(1, 2)
        # Flatten action token from (B, T_skipped, N) to (B, T_skipped*N)
        action_padding_mask = action_padding_mask.flatten(1, 2)

        # Cache from last rollout
        past_key_value = None
        if "decoder/cache" in input_dict:
            past_key_value = input_dict["decoder/cache"]

        if self.relative_pe:
            if use_cache:
                if "decoder/modeled_agent_position_history" not in input_dict:
                    input_dict["decoder/modeled_agent_position_history"] = input_dict["decoder/modeled_agent_position"
                                                                                      ].flatten(1, 2)
                    input_dict["decoder/modeled_agent_heading_history"] = input_dict["decoder/modeled_agent_heading"
                                                                                     ].flatten(1, 2)
                    input_dict["decoder/modeled_agent_valid_mask_history"] = action_padding_mask
                else:
                    input_dict["decoder/modeled_agent_position_history"] = torch.cat(
                        [
                            input_dict["decoder/modeled_agent_position_history"],
                            input_dict["decoder/modeled_agent_position"].flatten(1, 2),
                        ],
                        dim=1
                    )
                    input_dict["decoder/modeled_agent_heading_history"] = torch.cat(
                        [
                            input_dict["decoder/modeled_agent_heading_history"],
                            input_dict["decoder/modeled_agent_heading"].flatten(1, 2),
                        ],
                        dim=1
                    )
                    input_dict["decoder/modeled_agent_valid_mask_history"] = torch.cat(
                        [
                            input_dict["decoder/modeled_agent_valid_mask_history"],
                            action_padding_mask,
                        ], dim=1
                    )
                full_tgt_pos = input_dict["decoder/modeled_agent_position_history"]
                full_tgt_heading = input_dict["decoder/modeled_agent_heading_history"]
                full_tgt_mask = input_dict["decoder/modeled_agent_valid_mask_history"]
                # full_tgt_causal_mask = input_dict["decoder/modeled_agent_causal_mask_history"]
            else:
                full_tgt_pos = input_dict["decoder/modeled_agent_position"].flatten(1, 2)
                full_tgt_heading = input_dict["decoder/modeled_agent_heading"].flatten(1, 2)
                full_tgt_mask = action_padding_mask
                # full_tgt_causal_mask = action_casual_mask
        else:
            full_tgt_pos = None
            full_tgt_heading = None
            full_tgt_mask = None
            # full_tgt_causal_mask = None

        # === Call models ===
        decoded_tokens = self.decoder(
            tgt=action_token.swapaxes(0, 1),
            tgt_mask=action_casual_mask,  # swapaxes(0, 1),
            tgt_key_padding_mask=action_padding_mask,
            tgt_is_causal=True,
            tgt_pos=input_dict["decoder/modeled_agent_position"].flatten(1, 2),
            tgt_heading=input_dict["decoder/modeled_agent_heading"].flatten(1, 2),
            full_tgt_pos=full_tgt_pos,
            full_tgt_heading=full_tgt_heading,
            full_tgt_mask=full_tgt_mask,
            # full_tgt_causal_mask=full_tgt_causal_mask,
            memory=scene_token.swapaxes(0, 1),
            memory_mask=None,  # The casual mask for memory
            memory_key_padding_mask=scene_padding_mask,
            memory_is_causal=False,
            memory_pos=input_dict["encoder/scenario_position"],
            memory_heading=input_dict["encoder/scenario_heading"],
            past_key_value=past_key_value,
            use_cache=use_cache
        )

        if use_cache:
            decoded_tokens, past_key_value = decoded_tokens
            input_dict["decoder/cache"] = past_key_value

        decoded_tokens = decoded_tokens.swapaxes(0, 1)
        logits = unwrap(self.prediction_head(decoded_tokens[~action_padding_mask]), ~action_padding_mask)
        logits = logits.reshape(B, T_skipped, N, self.num_actions)

        # print("Input", input_action.shape)
        # # print("DECODE : ", decoded_tokens[~action_padding_mask][-62:].mean(0)[:5])
        # print("DECODED0", decoded_tokens.shape, decoded_tokens[0, :62].mean(-1)[:5])
        # print("DECODED-1", decoded_tokens.shape, decoded_tokens[0, -62:].mean(-1)[:5])
        # print("LOGIT:", logits[0, -1].mean(-1)[:5])
        # print("====")

        input_dict["decoder/output_logit"] = logits

        return input_dict


if __name__ == '__main__':
    from bmt.utils import debug_tools
    from bmt.models.scene_encoder import SceneEncoder

    config = debug_tools.get_debug_config()
    enc = SceneEncoder(config)
    dec = MotionDecoder(config)
    input_dict = debug_tools.get_debug_data()
    out = dec(enc(input_dict))
    print(out)
