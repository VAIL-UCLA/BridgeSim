import copy
import numbers
from typing import Optional, Callable, List
from typing import Union, Tuple

import torch
from torch import Tensor, Size
from torch import nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.transformer import Dropout, Linear
from torch.nn.parameter import Parameter

from bmt.models import relation
from bmt.models.layers.multi_head_attention import MultiheadAttention

_shape_t = Union[int, List[int], Size]


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _generate_square_subsequent_mask(
    sz: int,
    device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
    dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """

    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape, )  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ['norm']

    def __init__(
        self,
        decoder_layer,
        num_layers,
        d_model,
        self_attention_knn,
        cross_attention_knn,
        norm=None,
        relative_pe=False
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.relative_pe = relative_pe
        self.d_model = d_model
        self.self_attention_knn = self_attention_knn
        self.cross_attention_knn = cross_attention_knn

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_pos: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        tgt_heading: Optional[Tensor] = None,
        memory_heading: Optional[Tensor] = None,
        full_tgt_pos: Optional[Tensor] = None,
        full_tgt_heading: Optional[Tensor] = None,
        full_tgt_mask: Optional[Tensor] = None,
        # full_tgt_causal_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
        past_key_value=None,
        use_cache=False
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        rel_pe_self, rel_mask_self, rel_indices_self = None, None, None
        rel_pe_cross, rel_mask_cross, rel_indices_cross = None, None, None
        if self.relative_pe:
            rel_pe_cross, rel_mask_cross, rel_indices_cross = relation.compute_relation(
                query_pos=tgt_pos,
                query_heading=tgt_heading,
                query_mask=tgt_key_padding_mask,
                key_pos=memory_pos,
                key_heading=memory_heading,
                key_mask=memory_key_padding_mask,
                hidden_dim=self.d_model,
                causal_mask=None,
                knn=self.cross_attention_knn
            )

            # PZH: This is very vulnerable to bugs as the query should attend to the q at this moment as well as all
            # history queries.
            if full_tgt_pos is None:
                causal_mask = tgt_mask
            else:
                # No need to consider causal mask if in autoregressive decoding.
                causal_mask = None
            rel_pe_self, rel_mask_self, rel_indices_self = relation.compute_relation(
                query_pos=tgt_pos,
                query_heading=tgt_heading,
                query_mask=tgt_key_padding_mask,
                key_pos=full_tgt_pos if full_tgt_pos is not None else tgt_pos,
                key_heading=full_tgt_heading if full_tgt_heading is not None else tgt_heading,
                key_mask=full_tgt_mask if full_tgt_mask is not None else tgt_key_padding_mask,
                hidden_dim=self.d_model,
                causal_mask=causal_mask,
                knn=self.self_attention_knn
            )
            # print("RELATION PE SIZE: ", rel_pe_self.shape)

        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        new_past_key_value = None
        if use_cache:
            new_past_key_value = ()
            tgt_is_causal = False
            assert memory_is_causal is False

        for layer_idx, mod in enumerate(self.layers):
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
                # past_key_value=past_key_value[layer_idx] if past_key_value else None
                past_key_value=past_key_value[layer_idx] if past_key_value else None,
                use_cache=use_cache,
                rel_pe_self=rel_pe_self,
                rel_mask_self=rel_mask_self,
                rel_indices_self=rel_indices_self,
                rel_pe_cross=rel_pe_cross,
                rel_mask_cross=rel_mask_cross,
                rel_indices_cross=rel_indices_cross
            )

            if use_cache:
                output, new_past_key_value_layer = output
                new_past_key_value += (new_past_key_value_layer, )

        if self.norm is not None:
            output = self.norm(output)

        if use_cache:
            return output, new_past_key_value
        return output


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
    """

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
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        relative_pe=False,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            disable_projection=pre_projection,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs
        )
        self.multihead_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            disable_projection=pre_projection,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.pre_projection = pre_projection
        if pre_projection:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            # self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            # self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)

            self.ca_qcontent_proj = nn.Linear(d_model, d_model)
            # self.ca_qpos_proj = nn.Linear(d_model, d_model)
            self.ca_kcontent_proj = nn.Linear(d_model, d_model)
            # self.ca_kpos_proj = nn.Linear(d_model, d_model)
            self.ca_v_proj = nn.Linear(d_model, d_model)
            # self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.relative_pe = relative_pe
        if relative_pe:
            assert pre_projection is False, "Relative positional encoding is not supported with pre_projection"
            self.sa_relation_k = nn.Linear(3 * d_model, d_model)
            self.sa_relation_v = nn.Linear(3 * d_model, d_model)
            self.ca_relation_k = nn.Linear(3 * d_model, d_model)
            self.ca_relation_v = nn.Linear(3 * d_model, d_model)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        past_key_value=None,
        use_cache=False,
        rel_pe_self=None,
        rel_mask_self=None,
        rel_indices_self=None,
        rel_pe_cross=None,
        rel_mask_cross=None,
        rel_indices_cross=None
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
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
                past_key_value=past_key_value,
                use_cache=use_cache,
                rel_pe_self=rel_pe_self,
                rel_mask_self=rel_mask_self,
                rel_indices_self=rel_indices_self,
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
                past_key_value=None,
                use_cache=False,
                rel_pe_cross=rel_pe_cross,
                rel_mask_cross=rel_mask_cross,
                rel_indices_cross=rel_indices_cross
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            sa_out = self._sa_block(
                x,
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
                past_key_value=past_key_value,
                use_cache=use_cache,
                rel_pe_self=rel_pe_self,
                rel_mask_self=rel_mask_self,
                rel_indices_self=rel_indices_self,
            )
            x = self.norm1(x + sa_out)
            x = self.norm2(
                x + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                    past_key_value=None,
                    use_cache=False,
                    rel_pe_cross=rel_pe_cross,
                    rel_mask_cross=rel_mask_cross,
                    rel_indices_cross=rel_indices_cross
                )
            )
            x = self.norm3(x + self._ff_block(x))

        if use_cache:
            return x, self._new_self_key_value  # , self._new_cross_key_value)
        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        pos: Optional[Tensor] = None,
        is_causal: bool = False,
        past_key_value=None,
        use_cache=False,
        rel_pe_self=None,
        rel_mask_self=None,
        rel_indices_self=None,
    ) -> Tensor:

        if self.pre_projection:
            q = self.sa_qcontent_proj(x)
            # qpos = self.sa_qpos_proj(pos)
            # q = torch.cat([qcontent, qpos], dim=-1)

            k = self.sa_kcontent_proj(x)
            # kpos = self.sa_kpos_proj(pos)
            # k = torch.cat([kcontent, kpos], dim=-1)

            v = self.sa_v_proj(x)

        else:
            q = x
            k = x
            v = x

        if self.relative_pe:
            assert self.pre_projection is False
            relation_k = self.sa_relation_k(rel_pe_self)
            relation_v = self.sa_relation_v(rel_pe_self)
        else:
            relation_k = None
            relation_v = None

        x, _, new_key_value = self.self_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
            past_key_value=past_key_value,
            use_cache=use_cache,
            disable_projection=self.pre_projection,
            relation_k=relation_k if self.relative_pe else None,
            relation_v=relation_v if self.relative_pe else None,
            relation_mask=rel_mask_self if self.relative_pe else None,
            relation_indices=rel_indices_self if self.relative_pe else None,
        )
        self._new_self_key_value = new_key_value
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        pos: Optional[Tensor] = None,
        is_causal: bool = False,
        past_key_value=None,
        use_cache=False,
        rel_pe_cross=None,
        rel_mask_cross=None,
        rel_indices_cross=None
    ) -> Tensor:

        if self.pre_projection:
            q = self.ca_qcontent_proj(x)
            # qpos = self.sa_qpos_proj(pos)
            # q = torch.cat([qcontent, qpos], dim=-1)

            k = self.ca_kcontent_proj(mem)
            # kpos = self.sa_kpos_proj(pos)
            # k = torch.cat([kcontent, kpos], dim=-1)

            v = self.ca_v_proj(mem)

        else:
            q = x
            k = mem
            v = mem

        if self.relative_pe:
            assert self.pre_projection is False
            relation_k = self.sa_relation_k(rel_pe_cross)
            relation_v = self.sa_relation_v(rel_pe_cross)
        else:
            relation_k = None
            relation_v = None

        x, _, new_key_value = self.multihead_attn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
            past_key_value=past_key_value,
            use_cache=use_cache,
            disable_projection=self.pre_projection,
            relation_k=relation_k if self.relative_pe else None,
            relation_v=relation_v if self.relative_pe else None,
            relation_mask=rel_mask_cross if self.relative_pe else None,
            relation_indices=rel_indices_cross if self.relative_pe else None,
        )
        self._new_cross_key_value = new_key_value
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
