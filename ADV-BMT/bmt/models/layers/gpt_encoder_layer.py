import torch
import torch.nn as nn
from torch.nn import Module

from bmt.models.layers import common_layers
from bmt.models.layers.decoder_layer import _get_clones
from bmt.models.layers.gpt_decoder_layer import MultiheadAttentionLayer


class SelfAttTransformerEncoder(Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, scene_tokens, scene_info, edge_features, edge_features_v=None, block_mask=None):
        output = scene_tokens
        for layer_idx, mod in enumerate(self.layers):
            output, new_cache, edge_features, block_mask = mod(
                output, scene_info, edge_features, edge_features_v=edge_features_v, block_mask=block_mask
            )
        output = self.norm(output)
        return output


class SelfAttTransformerEncoderLayer(Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        simple_relation=False,
        simple_relation_factor=1,
        dropout=0.0,
        is_v7=False,
        update_relation=False,
        add_relation_to_v=None,
        remove_rel_norm=None
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.s2s_norm = nn.LayerNorm(d_model)

        self.remove_rel_norm = remove_rel_norm
        if not remove_rel_norm:
            self.s2s_norm_rel = nn.LayerNorm(d_model)
            if add_relation_to_v:
                self.s2s_norm_rel_v = nn.LayerNorm(d_model)

        self.cross_s2s = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=simple_relation,
            simple_relation_factor=simple_relation_factor,
            is_v7=True,
            add_relation_to_v=add_relation_to_v,
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = common_layers.Mlp(
            in_features=d_model,
            hidden_features=4 * d_model,
            act_layer=approx_gelu,
            drop=dropout,
        )
        self.mlp_prenorm = nn.LayerNorm(d_model)
        self.update_relation = update_relation
        self.add_relation_to_v = add_relation_to_v

        # self.mlp_rel = common_layers.Mlp(in_features=d_model, hidden_features=4 * d_model, act_layer=approx_gelu, drop=0, is_v7=is_v7)
        # self.mlp_rel_prenorm = nn.LayerNorm(d_model)

    def forward(self, scene_tokens, scene_info, edge_features, edge_features_v=None, block_mask=None):
        x = self.s2s_norm(scene_tokens)
        out, cache, edge_features_out = self.cross_s2s(
            q=x,
            k=x,
            edge_index=scene_info['edge_index'],
            edge_features=self.s2s_norm_rel(edge_features) if not self.remove_rel_norm else edge_features,
            edge_features_v=(self.s2s_norm_rel_v(edge_features_v) if not self.remove_rel_norm else edge_features_v)
            if edge_features_v is not None else None,
        )
        scene_tokens = scene_tokens + out
        out = self.mlp(self.mlp_prenorm(scene_tokens))
        scene_tokens = scene_tokens + out

        if self.update_relation:
            assert self.add_relation_to_v is False
            edge_features_out = edge_features_out + edge_features
        else:
            edge_features_out = edge_features
        return scene_tokens, cache, edge_features_out, block_mask  # , edge_features_v_out
