from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from bmt.models.layers import common_layers
from bmt.models.layers.decoder_layer import _get_clones
from bmt.utils import utils

# from torch.nn.attention.flex_attention import (
#     _DEFAULT_SPARSE_BLOCK_SIZE,
#     create_block_mask,
#     create_mask,
#     flex_attention,
#     _round_up_to_multiple
# )
# from torch.nn.attention.flex_attention import flex_attention, create_block_mask
# flex_attention = torch.compile(flex_attention)
# create_block_mask = torch.compile(create_block_mask, dynamic=False)
# create_block_mask = torch.compile(create_block_mask)


class MultiCrossAttTransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(
        self,
        decoder_layer,
        num_layers,
        d_model,
        norm=None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

    def forward(
        self,
        *,
        agent_token,
        scene_token,
        a2a_info=None,
        a2t_info=None,
        a2s_info=None,
        condition_token=None,
        past_key_value_list=None,
        use_cache=False
    ):
        new_past_key_value_list = []
        output = agent_token
        for layer_idx, mod in enumerate(self.layers):
            cache = past_key_value_list[layer_idx] if past_key_value_list is not None else None
            output, past_key_value = mod(
                agent_token=output,
                scene_token=scene_token,
                a2a_info=a2a_info,
                a2t_info=a2t_info,
                a2s_info=a2s_info,
                condition_token=condition_token,
                use_cache=use_cache,
                past_key_value=cache,
            )
            if use_cache:
                new_past_key_value_list.append(past_key_value)
        if self.norm is not None:
            output = self.norm(output)
        if use_cache:
            return output, new_past_key_value_list
        return output


class MultiheadAttentionLayer(MessagePassing):
    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.0,
        simple_relation=False,
        simple_relation_factor=2,
        is_v7=False,
        update_relation=False,
        add_relation_to_v=None
    ):
        super(MultiheadAttentionLayer, self).__init__(aggr='add', node_dim=0)  # Aggregation method 'add'
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert dropout == 0.0, "dropout is not supported"
        self.dropout = nn.Dropout(dropout)
        self.simple_relation = simple_relation
        if is_v7:
            if simple_relation:
                self.relation_head_dim = self.head_dim // simple_relation_factor
                self.to_q_relation = nn.Linear(d_model, d_model)
            self.to_k_r = nn.Linear(d_model // simple_relation_factor, d_model)
            self.to_v_r = nn.Linear(d_model // simple_relation_factor, d_model)
            self.to_k = nn.Linear(d_model, d_model)
            self.to_q = nn.Linear(d_model, d_model)
            self.to_v = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
            # self.out.weight.data.zero_()
        else:
            raise ValueError()
            if simple_relation:
                self.relation_head_dim = self.head_dim // simple_relation_factor
                self.to_q_relation = nn.Linear(d_model, d_model // simple_relation_factor)
            self.to_k = nn.Linear(d_model, d_model)
            self.to_q = nn.Linear(d_model, d_model)
            self.to_v = nn.Linear(d_model, d_model)
        self.is_v7 = is_v7
        self.update_relation = update_relation
        assert update_relation is False
        assert add_relation_to_v is not None, "add_relation_to_v is required."
        self.add_relation_to_v = add_relation_to_v
        # self.out_rel = nn.Linear(d_model, d_model, bias=False)
        # self.out_rel.weight.data.zero_()

    def forward(
        self,
        q,
        k,
        edge_index,
        edge_features,
        edge_features_v=None,
        use_cache=False,
        cache=None,  #Relation=None
    ):
        B, Lq, D = q.shape
        _, Lk, _ = k.shape

        # Compute linear projections
        x_dst = q
        x_src = k
        Q = self.to_q(x_dst).reshape(-1, self.n_heads * self.head_dim)
        K = self.to_k(x_src).reshape(-1, self.n_heads * self.head_dim)
        V = self.to_v(x_src).reshape(-1, self.n_heads * self.head_dim)

        if cache is not None:
            past_key = cache[0]
            past_value = cache[1]
            key_B, key_T = cache[2]

            K = K.reshape(key_B, -1, self.n_heads * self.head_dim)
            past_key = past_key.reshape(key_B, key_T, self.n_heads * self.head_dim)
            K = torch.cat((past_key, K), dim=1)
            K = K.reshape(-1, self.n_heads * self.head_dim)

            V = V.reshape(key_B, -1, self.n_heads * self.head_dim)
            past_value = past_value.reshape(key_B, key_T, self.n_heads * self.head_dim)
            V = torch.cat((past_value, V), dim=1)
            V = V.reshape(-1, self.n_heads * self.head_dim)

        assert edge_index[0].max() < K.shape[0], f"{edge_index[0].max()} >= {K.shape[0]}"
        assert edge_index[1].max() < Q.shape[0], f"{edge_index[1].max()} >= {Q.shape[0]}"

        if use_cache:
            new_cache = [K, V]
        else:
            new_cache = None

        if self.simple_relation:
            Q_relation = self.to_q_relation(x_dst).reshape(-1, self.n_heads * self.head_dim)
            Q = torch.cat([Q, Q_relation], dim=-1)

        if self.is_v7:
            if self.add_relation_to_v:
                assert edge_features_v is not None
            else:
                assert edge_features_v is None
                edge_features_v = edge_features
            edge_features = self.to_k_r(edge_features)
            edge_features_v = self.to_v_r(edge_features_v)

        # Propagate messages using edge_index
        out, new_edge_features = self.propagate(
            edge_index=edge_index,
            # x_dst=x_dst.reshape(-1, self.n_heads * self.head_dim),
            q=Q,
            k=K,
            v=V,
            edge_features=edge_features,
            edge_features_v=edge_features_v,
        )

        # Project the output back to original dimension
        out = out.reshape(B, Lq, D)
        if new_edge_features is not None:
            new_edge_features = new_edge_features.reshape(-1, D)
        if self.is_v7:
            out = self.out(out)
            # new_edge_features = self.out_rel(new_edge_features)
            return out, new_cache, new_edge_features  #, edge_features, edge_features_v

        return out, new_cache

    def message(
        self, q_i, k_j, v_j, edge_features, edge_features_v, index, ptr, edge_index, edge_index_i, edge_index_j,
        relation
    ):
        k_j = k_j.reshape(-1, self.n_heads, self.head_dim)
        v_j = v_j.reshape(-1, self.n_heads, self.head_dim)

        if edge_features is not None and not self.simple_relation:
            raise ValueError()
            edge_features = edge_features.reshape(-1, self.n_heads, self.head_dim)

            if self.is_v7:
                raise ValueError()

            # Compute relative positional encoding if enabled
            k_j = k_j + edge_features  # Add relative position embedding to Key

        if self.simple_relation:
            q_i, q_relation = q_i[:, :self.n_heads * self.head_dim], q_i[:, self.n_heads * self.head_dim:]
            # Compute attention scores
            q_i = q_i.reshape(-1, self.n_heads, self.head_dim)
            q_relation = q_relation.reshape(-1, self.n_heads, self.head_dim)

            edge_features = edge_features.reshape(-1, self.n_heads, self.head_dim)

            # if self.is_v7:
            #
            #     # Do the so-call QK norm here.
            #     # q_i = nn.functional.rms_norm(q_i, normalized_shape=(q_i.shape[-1], ))
            #     # q_relation = nn.functional.rms_norm(q_relation, normalized_shape=(q_relation.shape[-1], ))
            #     # k_j = nn.functional.rms_norm(k_j, normalized_shape=(k_j.shape[-1], ))
            #     # edge_features = nn.functional.rms_norm(edge_features, normalized_shape=(edge_features.shape[-1], ))

            attn_scores = (q_i * k_j).sum(dim=-1) / self.head_dim**0.5  # Scaled dot-product
            attn_scores_relation = (q_relation * edge_features).sum(dim=-1) / self.head_dim**0.5
            attn_scores = attn_scores + attn_scores_relation

        else:
            q_i = q_i.reshape(-1, self.n_heads, self.head_dim)
            # Compute attention scores
            attn_scores = (q_i * k_j).sum(dim=-1) / self.head_dim**0.5  # Scaled dot-product

        attn_weights = softmax(attn_scores, index=index, ptr=ptr)
        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights

        if edge_features_v is not None:
            edge_features_v = edge_features_v.reshape(-1, self.n_heads, self.head_dim)

            v_j = v_j + edge_features_v

        if self.update_relation:
            new_edge_features = edge_features + edge_features_v
        else:
            new_edge_features = None

        attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights

        return v_j * attn_weights.unsqueeze(-1), new_edge_features

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tensor:
        raw_inputs, new_edge_features = inputs
        inputs = super().aggregate(raw_inputs, index, ptr, dim_size)
        if new_edge_features is not None:
            new_edge_features = new_edge_features + raw_inputs
        return inputs, new_edge_features


class MultiheadAttentionLayerWithFlex(MessagePassing):
    def __init__(self, d_model, n_heads, dropout=0.1, simple_relation=False, simple_relation_factor=2, is_v7=False):
        super(MultiheadAttentionLayerWithFlex, self).__init__(aggr='add', node_dim=0)  # Aggregation method 'add'
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.relation_head_dim = self.head_dim // simple_relation_factor
        self.to_q_relation = nn.Linear(d_model, d_model, bias=False)
        self.to_k_r = nn.Linear(d_model // simple_relation_factor, d_model, bias=False)
        self.to_v_r = nn.Linear(d_model // simple_relation_factor, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.out.weight.data.zero_()

    def forward(self, q, k, qk_valid_mask, relation, block_mask=None, relation_v=None, use_cache=False, cache=None):

        # TODO: assert relation shape

        B, Lq, D = q.shape
        _, Lk, _ = k.shape

        # Compute linear projections
        x_dst = q
        x_src = k
        Q = self.to_q(x_dst).reshape(-1, self.n_heads * self.head_dim)
        K = self.to_k(x_src).reshape(-1, self.n_heads * self.head_dim)
        V = self.to_v(x_src).reshape(-1, self.n_heads * self.head_dim)

        if cache is not None:
            past_key = cache[0]
            past_value = cache[1]
            key_B, key_T = cache[2]

            K = K.reshape(key_B, -1, self.n_heads * self.head_dim)
            past_key = past_key.reshape(key_B, key_T, self.n_heads * self.head_dim)
            K = torch.cat((past_key, K), dim=1)
            K = K.reshape(-1, self.n_heads * self.head_dim)

            V = V.reshape(key_B, -1, self.n_heads * self.head_dim)
            past_value = past_value.reshape(key_B, key_T, self.n_heads * self.head_dim)
            V = torch.cat((past_value, V), dim=1)
            V = V.reshape(-1, self.n_heads * self.head_dim)

        # assert edge_index[0].max() < K.shape[0], f"{edge_index[0].max()} >= {K.shape[0]}"
        # assert edge_index[1].max() < Q.shape[0], f"{edge_index[1].max()} >= {Q.shape[0]}"

        if use_cache:
            new_cache = [K, V]
        else:
            new_cache = None

        # newB, newLq, _ = x.shape
        # qk_valid_mask = a2t_info["attn_valid_mask"]

        _, _, newLk = qk_valid_mask.shape

        # key = out
        # value = out

        K = K.reshape(B, 1, Lk, D)
        # rel = a2t_info['relation']  # newB, newLq, newLk, 128
        K = K + relation

        V = V.reshape(B, 1, newLk, D)
        # rel_v = a2t_info['relation_v']  # newB, newLq, newLk, 128
        # value = value + rel_v
        assert relation_v is None
        relation_v = relation
        V = V + relation_v

        Lk_new = Lk * Lq

        # TODO: in future update the swapaxes.
        K = K.reshape(B, Lk_new, self.n_heads, self.head_dim).swapaxes(1, 2)
        V = V.reshape(B, Lk_new, self.n_heads, self.head_dim).swapaxes(1, 2)
        Q = Q.reshape(B, Lq, self.n_heads, self.head_dim).swapaxes(1, 2)

        if block_mask is None:
            qk_valid_mask = qk_valid_mask.reshape(B, Lq, 1, newLk).expand(B, Lq, Lq, newLk).reshape(B, Lq, Lk_new)

            # TODO: How to select?
            block_size = _DEFAULT_SPARSE_BLOCK_SIZE
            # block_size = 4

            Lq_padded = _round_up_to_multiple(Lq, block_size)
            Lk_padded = _round_up_to_multiple(Lk_new, block_size)
            new_valid_mask = qk_valid_mask.new_zeros(B, Lq_padded, Lk_padded)
            new_valid_mask[:, :Lq, :Lk_new] = qk_valid_mask

            # TODO: Make the mask before!
            # TODO: Can implement the sliding window here.
            def mask_mod(b, h, q_idx, kv_idx):
                realq = kv_idx // newLk
                m1 = q_idx == realq
                # FIXME
                # FIXME
                m3 = new_valid_mask[b, q_idx, kv_idx]
                return m1 & m3
                # return m1

            # res = []
            # import numpy as np
            # for q in range(Lq_padded):
            #     res.append([mask_mod(0, 0, q, v).item() for v in range(Lk_padded)])
            # res = np.array(res)

            block_mask = create_block_mask(
                mask_mod=mask_mod,
                B=B,
                H=self.n_heads,
                Q_LEN=Lq_padded,
                KV_LEN=Lk_padded,
                device=Q.device,
                BLOCK_SIZE=block_size,
                # _compile=True
            )

        flex_out = flex_attention(
            query=Q,
            key=K,
            value=V,
            block_mask=block_mask,
        )

        # # if self.simple_relation:
        # Q_relation = self.to_q_relation(x_dst).reshape(-1, self.n_heads * self.head_dim)
        # Q = torch.cat([Q, Q_relation], dim=-1)
        #
        #
        # # Propagate messages using edge_index
        # out, new_edge_features = self.propagate(
        #     edge_index=edge_index,
        #     x_dst=x_dst.reshape(-1, self.n_heads * self.head_dim),
        #     q=Q,
        #     k=K,
        #     v=V,
        #     edge_features=edge_features,
        #     edge_features_v=edge_features_v,
        # )

        # Project the output back to original dimension
        out = flex_out.reshape(B, Lq, D)
        # new_edge_features = new_edge_features.reshape(-1, D)
        # if self.is_v7:
        out = self.out(out)
        # new_edge_features = self.out_rel(new_edge_features)
        new_edge_features = None
        return out, new_cache, new_edge_features, block_mask  #, edge_features, edge_features_v

        # return out, new_cache

    # def message(
    #     self, q_i, k_j, v_j, edge_features, edge_features_v, index, ptr, edge_index, edge_index_i, edge_index_j,
    #     relation
    # ):
    #     k_j = k_j.reshape(-1, self.n_heads, self.head_dim)
    #     v_j = v_j.reshape(-1, self.n_heads, self.head_dim)
    #
    #     q_i, q_relation = q_i[:, :self.n_heads * self.head_dim], q_i[:, self.n_heads * self.head_dim:]
    #     # Compute attention scores
    #     q_i = q_i.reshape(-1, self.n_heads, self.head_dim)
    #     q_relation = q_relation.reshape(-1, self.n_heads, self.head_dim)
    #
    #
    #     edge_features = edge_features.reshape(-1, self.n_heads, self.head_dim)
    #
    #     if self.is_v7:
    #
    #         # Do the so-call QK norm here.
    #         q_i = nn.functional.rms_norm(q_i, normalized_shape=(q_i.shape[-1], ))
    #         q_relation = nn.functional.rms_norm(q_relation, normalized_shape=(q_relation.shape[-1], ))
    #         k_j = nn.functional.rms_norm(k_j, normalized_shape=(k_j.shape[-1], ))
    #         edge_features = nn.functional.rms_norm(edge_features, normalized_shape=(edge_features.shape[-1], ))
    #
    #     attn_scores = (q_i * k_j + q_relation * edge_features).sum(dim=-1) / self.head_dim**0.5  # Scaled dot-product
    #     # attn_scores_relation = (q_relation * edge_features).sum(dim=-1) / self.head_dim**0.5
    #     # attn_scores = attn_scores + attn_scores_relation
    #
    #
    #     attn_weights = softmax(attn_scores, index=index, ptr=ptr)
    #     attn_weights = self.dropout(attn_weights)  # Apply dropout to attention weights
    #
    #     if edge_features_v is not None:
    #         edge_features_v = edge_features_v.reshape(-1, self.n_heads, self.head_dim)
    #
    #         v_j = v_j + edge_features_v
    #
    #     new_edge_features = edge_features + edge_features_v
    #
    #     return v_j * attn_weights.unsqueeze(-1), new_edge_features
    #
    # def aggregate(
    #     self,
    #     inputs: Tensor,
    #     index: Tensor,
    #     ptr: Optional[Tensor] = None,
    #     dim_size: Optional[int] = None,
    # ) -> Tensor:
    #     raw_inputs, new_edge_features = inputs
    #     inputs = super().aggregate(raw_inputs, index, ptr, dim_size)
    #     new_edge_features = new_edge_features + raw_inputs
    #     return inputs, new_edge_features


class MultiCrossAttTransformerDecoderLayer(Module):
    __constants__ = ['norm_first']

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        use_adaln=False,
        simple_relation=False,
        simple_relation_factor=None,
        is_v7=False,
        update_relation=False,
        add_relation_to_v=None,
        remove_rel_norm=None,
    ) -> None:
        super().__init__()
        self.cross_a2t = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=simple_relation,
            simple_relation_factor=simple_relation_factor,
            is_v7=is_v7,
            update_relation=update_relation,
            add_relation_to_v=add_relation_to_v,
        )
        self.cross_a2a = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=simple_relation,
            simple_relation_factor=simple_relation_factor,
            is_v7=is_v7,
            update_relation=update_relation,
            add_relation_to_v=add_relation_to_v,
        )
        self.cross_a2s = MultiheadAttentionLayer(
            d_model=d_model,
            n_heads=nhead,
            dropout=dropout,
            simple_relation=simple_relation,
            simple_relation_factor=simple_relation_factor,
            is_v7=is_v7,
            update_relation=update_relation,
            add_relation_to_v=add_relation_to_v,
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = common_layers.Mlp(
            in_features=d_model, hidden_features=4 * d_model, act_layer=approx_gelu, drop=dropout, is_v7=is_v7
        )

        self.use_adaln = use_adaln
        if use_adaln:
            self.a2t_adaln_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.a2a_adaln_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.a2s_adaln_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.mlp_adaln_prenorm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            # https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L113
            self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 12 * d_model, bias=True))
            # self.adaLN_modulation_gate = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 4 * d_model, bias=True))
        else:
            self.a2s_norm = nn.LayerNorm(d_model)
            self.a2t_norm = nn.LayerNorm(d_model)
            self.a2a_norm = nn.LayerNorm(d_model)
            self.mlp_prenorm = nn.LayerNorm(d_model)

        self.remove_rel_norm = remove_rel_norm
        assert remove_rel_norm is not None
        if not remove_rel_norm:
            self.a2t_norm_rel = nn.LayerNorm(d_model)
            self.a2a_norm_rel = nn.LayerNorm(d_model)
            self.a2s_norm_rel = nn.LayerNorm(d_model)

        self.update_relation = update_relation
        self.add_relation_to_v = add_relation_to_v
        assert add_relation_to_v is not None
        if add_relation_to_v and (not remove_rel_norm):
            assert update_relation is False
            self.a2t_norm_rel_v = nn.LayerNorm(d_model)
            self.a2a_norm_rel_v = nn.LayerNorm(d_model)
            self.a2s_norm_rel_v = nn.LayerNorm(d_model)
        if update_relation:
            assert add_relation_to_v is False

    # def __setstate__(self, state):
    #     # if 'activation' not in state:
    #     #     state['activation'] = F.relu
    #     super().__setstate__(state)

    def forward(
        self,
        *,
        agent_token,
        scene_token,
        a2a_info,
        a2t_info,
        a2s_info,
        condition_token,
        use_cache=False,
        past_key_value=None
    ):
        B, T, N, D = agent_token.shape
        x = agent_token

        if self.use_adaln:
            assert condition_token.ndim == agent_token.ndim  # (B, T, N, D)
            adaln_params = self.adaln_modulation(condition_token)
            adaln_params = adaln_params.expand(B, T, N, -1)
            adaln_params = adaln_params.chunk(12, dim=-1)
            shift_a2t, scale_a2t, gate_a2t = adaln_params[:3]
            shift_a2a, scale_a2a, gate_a2a = adaln_params[3:6]
            shift_a2s, scale_a2s, gate_a2s = adaln_params[6:9]
            shift_ff, scale_ff, gate_ff = adaln_params[9:12]

        # === agent-temporal attention ===
        # B,T,N,D -> BN, T, D
        x = x.swapaxes(1, 2).flatten(0, 1)
        out = x
        if self.use_adaln:
            out = self.a2t_adaln_norm(out)
            out = utils.modulate(out, shift_a2t.swapaxes(1, 2).flatten(0, 1), scale_a2t.swapaxes(1, 2).flatten(0, 1))
        else:
            out = self.a2t_norm(out)

        a2t_rel = a2t_info['edge_features']
        if self.remove_rel_norm:
            a2t_rel_out = a2t_rel
            a2t_rel_out_v = a2t_info['edge_features_v'] if self.add_relation_to_v else None
        else:
            a2t_rel_out = self.a2t_norm_rel(a2t_rel)
            a2t_rel_out_v = self.a2t_norm_rel_v(a2t_info['edge_features_v']) if self.add_relation_to_v else None
        # if "block_mask" not in a2t_info:
        # a2t_info["block_mask"] = None
        # out, past_key_value_a2t, a2t_rel_out, a2t_block_mask = self.cross_a2t(
        out, past_key_value_a2t, a2t_rel_out = self.cross_a2t(
            q=out,
            k=out,
            edge_features=a2t_rel_out,
            edge_features_v=a2t_rel_out_v,
            edge_index=a2t_info['edge_index'],
            # qk_valid_mask=a2t_info["attn_valid_mask"],
            # relation=a2t_info["relation"],
            use_cache=use_cache,
            cache=past_key_value,
            # block_mask=a2t_info["block_mask"],
            # Relation=a2t_info["relation"]
        )
        assert out.shape == (B * N, T, D)
        if self.use_adaln:
            out = out * gate_a2t.swapaxes(1, 2).flatten(0, 1)
        x = x + out
        x = x.reshape(B, N, T, D).swapaxes(1, 2)
        if self.update_relation:
            a2t_rel_out = a2t_rel_out + a2t_rel
            a2t_info['edge_features'] = a2t_rel_out
            assert self.add_relation_to_v is False

        # === agent-agent attention ===
        x = x.reshape(B * T, N, D)
        out = x
        if self.use_adaln:
            out = self.a2a_adaln_norm(out)
            out = utils.modulate(out, shift_a2a.reshape(B * T, N, D), scale_a2a.reshape(B * T, N, D))
        else:
            out = self.a2a_norm(out)

        a2a_rel = a2a_info['edge_features']
        if self.remove_rel_norm:
            a2a_rel_out = a2a_rel
            a2a_rel_out_v = a2a_info['edge_features_v'] if self.add_relation_to_v else None
        else:
            a2a_rel_out = self.a2a_norm_rel(a2a_rel)
            a2a_rel_out_v = self.a2a_norm_rel_v(a2a_info['edge_features_v']) if self.add_relation_to_v else None

        if "block_mask" not in a2a_info:
            a2a_info["block_mask"] = None
        # out, _, a2a_rel_out, a2a_block_mask = self.cross_a2a(
        out, _, a2a_rel_out = self.cross_a2a(
            q=out,
            k=out,
            # qk_valid_mask=a2a_info["attn_valid_mask"],
            # relation=a2a_info["relation"],
            # block_mask=a2a_info["block_mask"],
            edge_features=a2a_rel_out,
            edge_features_v=a2a_rel_out_v,
            edge_index=a2a_info['edge_index'],
        )
        # a2a_info["block_mask"] = a2a_block_mask
        if self.use_adaln:
            out = out * gate_a2a.reshape(B * T, N, D)
        x = x + out
        x = x.reshape(B, T, N, D)
        if self.update_relation:
            a2a_rel_out = a2a_rel_out + a2a_rel
            a2a_info['edge_features'] = a2a_rel_out

        # === agent-scene attention ===
        x = x.reshape(B, T * N, D)
        out = x
        if self.use_adaln:
            out = self.a2s_adaln_norm(out)
            out = utils.modulate(out, shift_a2s.reshape(B, T * N, D), scale_a2s.reshape(B, T * N, D))
        else:
            out = self.a2s_norm(out)

        a2s_rel = a2s_info['edge_features']
        if self.remove_rel_norm:
            a2s_rel_out = a2s_rel
            a2s_rel_out_v = a2s_info['edge_features_v'] if self.add_relation_to_v else None
        else:
            a2s_rel_out = self.a2s_norm_rel(a2s_rel)
            a2s_rel_out_v = self.a2s_norm_rel_v(a2s_info['edge_features_v']) if self.add_relation_to_v else None

        if "block_mask" not in a2s_info:
            a2s_info["block_mask"] = None
        # out, _, a2s_rel_out, a2s_block_mask = self.cross_a2s(
        out, _, a2s_rel_out = self.cross_a2s(
            q=out,
            k=scene_token,
            # qk_valid_mask=a2s_info["attn_valid_mask"],
            # relation=a2s_info["relation"],
            # block_mask=a2s_info["block_mask"],
            edge_features=a2s_rel_out,
            edge_features_v=a2s_rel_out_v,
            edge_index=a2s_info['edge_index'],
        )
        # a2s_info["block_mask"] = a2s_block_mask
        if self.use_adaln:
            out = out * gate_a2s.reshape(B, T * N, D)
        x = x + out
        x = x.reshape(B, T, N, D)
        if self.update_relation:
            a2s_rel_out = a2s_rel_out + a2s_rel
            a2s_info['edge_features'] = a2s_rel_out

        # Print to make sure overwriting dict is valid.
        # print("a2s_rel_out", a2s_rel.mean().item(), a2s_rel.std().item())

        # === Feed-forward layer ===
        out = x
        if self.use_adaln:
            out = self.mlp_adaln_prenorm(out)
            out = utils.modulate(out, shift_ff, scale_ff)
        else:
            out = self.mlp_prenorm(out)
        out = self.mlp(out)
        if self.use_adaln:
            out = out * gate_ff
        x = x + out

        return x, past_key_value_a2t
