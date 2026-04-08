from typing import Optional, Callable
from typing import Union

from torch import Tensor
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.modules.transformer import Dropout, Linear

from bmt.models.layers.decoder_layer import LayerNorm, _get_activation_fn
from bmt.models.layers.multi_head_attention import MultiheadAttention


class TransformerEncoderLayer(Module):
    __constants__ = ['norm_first']

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        pre_projection: bool = False,
        relative_pe: bool = False,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # if pre_projection:
        #     kdim = d_model * 2
        # else:
        #     kdim = d_model
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            kdim=d_model,
            vdim=d_model,
            disable_projection=pre_projection,
            **factory_kwargs
        )
        # self.multihead_attn = MultiheadAttention(
        #     d_model, nhead, dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
        # )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.pre_projection = pre_projection
        if pre_projection:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.sa_vpos_proj = nn.Linear(d_model, d_model)

        self.relative_pe = relative_pe
        if relative_pe:
            assert pre_projection is False, "Relative positional encoding is not supported with pre_projection"
            self.sa_relation_k = nn.Linear(3 * d_model, d_model)
            self.sa_relation_v = nn.Linear(3 * d_model, d_model)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        pos: Optional[Tensor] = None,
        relation: Optional[Tensor] = None,
        relation_mask: Optional[Tensor] = None,
        relation_indices: Optional[Tensor] = None,
        # past_key_value=None,
        # use_cache=False
    ) -> Tensor:
        # Split past key and value states for self-attention and multi-head attention
        # past_self_key_value = past_key_value[0] if past_key_value is not None else None
        # past_cross_key_value = past_key_value[1] if past_key_value is not None else None

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
                pos=pos,
                relation=relation,
                relation_mask=relation_mask,
                # past_key_value=past_key_value,
                # use_cache=use_cache
            )
            # x = x + self._mha_block(
            #     self.norm2(x),
            #     memory,
            #     memory_mask,
            #     memory_key_padding_mask,
            #     memory_is_causal,
            #     past_key_value=None,
            #     use_cache=False
            # )
            x = x + self._ff_block(self.norm2(x))
        else:
            sa_out = self._sa_block(
                x,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
                is_causal=tgt_is_causal,
                pos=pos,
                relation=relation,
                relation_mask=relation_mask,
                relation_indices=relation_indices,
                # past_key_value=past_key_value,
                # use_cache=use_cache
            )
            x = self.norm1(x + sa_out)
            # x = self.norm2(
            #     x + self._mha_block(
            #         x,
            #         memory,
            #         memory_mask,
            #         memory_key_padding_mask,
            #         memory_is_causal,
            #         past_key_value=None,
            #         use_cache=False
            #     )
            # )
            x = self.norm2(x + self._ff_block(x))

        # if use_cache:
        #     return x, self._new_self_key_value  # , self._new_cross_key_value)
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        pos: Optional[Tensor] = None,
        relation: Optional[Tensor] = None,
        relation_mask: Optional[Tensor] = None,
        relation_indices: Optional[Tensor] = None,
        is_causal: bool = False,
        # past_key_value=None,
        # use_cache=False
    ) -> Tensor:

        if self.pre_projection:
            q = self.sa_qcontent_proj(x)
            k = self.sa_kcontent_proj(x)
            v = self.sa_v_proj(x)
            qpos = self.sa_qpos_proj(pos)
            kpos = self.sa_kpos_proj(pos)
            vpos = self.sa_vpos_proj(pos)

        else:
            q = x
            k = x
            v = x
            qpos = None
            kpos = None
            vpos = None

        if self.relative_pe:
            assert self.pre_projection is False
            relation_k = self.sa_relation_k(relation)
            relation_v = self.sa_relation_v(relation)
            # relation_v = relation_k = relation
        else:
            relation_k = None
            relation_v = None

        # B, L, D = q.shape
        # if attn_mask is None:
        #     attn_mask = q.new_zeros((B, L, L))

        x, _, new_key_value = self.self_attn(
            q,
            k,
            v,
            query_pos=qpos,
            key_pos=kpos,
            value_pos=vpos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            relation_k=relation_k if self.relative_pe else None,
            relation_v=relation_v if self.relative_pe else None,
            relation_mask=relation_mask if self.relative_pe else None,
            relation_indices=relation_indices if self.relative_pe else None,
            need_weights=True if relation is not None else False,
            disable_projection=True if self.pre_projection else False,
            # past_key_value=past_key_value,
            # use_cache=use_cache
        )
        # self._new_self_key_value = new_key_value
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
