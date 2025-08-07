import copy
from typing import Any, Dict, Optional, Tuple, cast

import torch
from torch import Tensor
from torch.nn import LayerNorm, Module, ModuleList, ReLU
from torch.nn import functional as F  # noqa: N812
from torch.nn.modules.transformer import _detect_is_causal_mask, _get_seq_len

from .base import VIModule
from .linear import VILinear
from .priors import MeanFieldNormalPrior, Prior
from .sequential import VIResidualConnection
from .utils.common_types import VIkwargs, VIReturn, _prior_any_t, _vardist_any_t
from .variational_distributions import MeanFieldNormalVarDist, VariationalDistribution


class VIMultiheadAttention(VIModule):
    """
    Allows the model to jointly attend to information from different representation subspaces.

    Equivalent of :class:`nn.MultiheadAttention` with variational inference. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>`__
    for usage.

    This does not support the ``dropout`` argument.
    In addition to all other arguments, this class accepts :class:`~.VIkwargs`.

    This module's random variables are:

    - ("in_proj_weight", "out_proj_weight") if ``kdim`` and ``vdim`` are ``None`` or
      equal to ``embed_dim``.
    - ("q_proj_weight", "k_proj_weight", "v_proj_weight", "out_proj_weight") else.

    Additional random variables are appended in the following order based on the given
    arguments:

    - ("in_proj_bias", "out_proj_bias") if ``bias`` is ``True``.
    - ("bias_k", "bias_v") if ``add_bias_kv`` is ``True``.

    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        rescale_prior: bool = True,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            rescale_prior=rescale_prior,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (self.kdim == embed_dim) and (self.vdim == embed_dim)

        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        variables: Dict[str, Tuple[int, ...]] = dict()
        if not self._qkv_same_embed_dim:
            variables["q_proj_weight"] = (embed_dim, embed_dim)
            variables["k_proj_weight"] = (embed_dim, self.kdim)
            variables["v_proj_weight"] = (embed_dim, self.vdim)
        else:
            variables["in_proj_weight"] = (3 * embed_dim, embed_dim)
        variables["out_proj_weight"] = (embed_dim, embed_dim)
        if bias:
            variables["in_proj_bias"] = (3 * embed_dim,)
            variables["out_proj_bias"] = (embed_dim,)

        if add_bias_kv:
            variables["bias_k"] = (1, 1, embed_dim)
            variables["bias_v"] = (1, 1, embed_dim)

        self.add_kv_bias = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.bias = bias

        super().__init__(variables, **vikwargs)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        # need_weights: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> VIReturn[Tuple[Tensor, Optional[Tensor]]]:
        """
        Compute attention outputs using query, key, and value embeddings.

        Supports optional parameters for padding, masks and attention weights. See
        `documentation <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>`__
        of :class:`nn.MultiheadAttention` for details. For technical reasons
        ``need_weights`` is hardcoded as ``True``.

        This implementation also currently does not support the torch fastpath.
        """
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # This transposition scheme is copied from torch including the comment
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        params = self.sample_variables()
        if self.add_kv_bias:
            bias_v = params.pop()
            bias_k = params.pop()
        else:
            bias_k = bias_v = None

        if not self._qkv_same_embed_dim:
            if self.bias:
                (
                    q_proj_weight,
                    k_proj_weight,
                    v_proj_weight,
                    out_proj_weight,
                    in_proj_bias,
                    out_proj_bias,
                ) = params
            else:
                q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight = params
                in_proj_bias = out_proj_bias = None
            in_proj_weight = None

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k=bias_k,
                bias_v=bias_v,
                add_zero_attn=self.add_zero_attn,
                dropout_p=0.0,
                out_proj_weight=out_proj_weight,
                out_proj_bias=out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            if self.bias:
                in_proj_weight, out_proj_weight, in_proj_bias, out_proj_bias = params
            else:
                in_proj_weight, out_proj_weight = params
                in_proj_bias = out_proj_bias = None

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k=bias_k,
                bias_v=bias_v,
                add_zero_attn=self.add_zero_attn,
                dropout_p=0.0,
                out_proj_weight=out_proj_weight,
                out_proj_bias=out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        if self._return_log_probs:
            log_probs = self.get_log_probs(params)
            return (attn_output, attn_output_weights), log_probs
        else:
            return attn_output, attn_output_weights


class VITransformerEncoderLayer(VIModule):
    """
    TransformerEncoderLayer is made up of self-attn and feedforward network.

    Equivalent of :class:`nn.TransformerEncoderLayer` with variational inference. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html>`__
    for usage.

    This does not support the ``dropout`` argument.
    In addition to all other arguments, this class accepts :class:`~.VIkwargs`.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        activation: Module = ReLU(),
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        variational_distribution: VariationalDistribution = MeanFieldNormalVarDist(),
        prior: Prior = MeanFieldNormalPrior(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )
        super().__init__()
        self.self_attn = VIMultiheadAttention(
            d_model, nhead, batch_first=batch_first, bias=bias, **vikwargs
        )
        # Feedforward model
        self._ff_block = VIResidualConnection(
            VILinear(d_model, dim_feedforward, bias=bias, **vikwargs),
            activation,
            VILinear(dim_feedforward, d_model, bias=bias, **vikwargs),
        )

        # layer norms are treated non-Bayesian
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self._return_log_probs = return_log_probs

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> VIReturn[Tensor]:
        """
        Pass the input through the encoder layer.

        See `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TranformerEncoderLayer.html>`__
        of :class:`nn.TranformerEncoderLayer` for details.

        This implementation also currently does not support the torch fastpath.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        x = src
        # _ff_block already includes residual connection
        if self._return_log_probs and self.norm_first:
            (x1, _), lps1 = self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x2 = x + x1
            x3, lps2 = self._ff_block(self.norm2(x2))
            x4 = x3

            lps1, lps2 = cast(Tuple[Tensor, Tensor], (lps1, lps2))
            log_probs = lps1 + lps2
            return x4, log_probs
        elif self._return_log_probs:
            (x1, _), lps1 = self._sa_block(
                x, src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x2 = x + x1
            x3, lps2 = self._ff_block(self.norm1(x2))
            x4 = self.norm2(x3)

            lps1, lps2 = cast(Tuple[Tensor, Tensor], (lps1, lps2))
            log_probs = lps1 + lps2
            return x4, log_probs
        elif self.norm_first:
            x = (
                x
                + self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
                )[0]
            )
            x = self._ff_block(self.norm2(x))
            return x
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, src_mask, src_key_padding_mask, is_causal=is_causal
                )[0]
            )
            x = self.norm2(self._ff_block(x))
            return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> VIReturn[Tuple[Tensor, Optional[Tensor]]]:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return x


class VITransformerDecoderLayer(VIModule):
    """
    TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Equivalent of :class:`nn.TransformerDecoderLayer` with variational inference. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html>`__
    for usage.

    This does not support the ``dropout`` argument.
    In addition to all other arguments, this class accepts :class:`~.VIkwargs`.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        activation: Module = ReLU(),
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        batch_first: bool = True,
        bias: bool = True,
        variational_distribution: VariationalDistribution = MeanFieldNormalVarDist(),
        prior: Prior = MeanFieldNormalPrior(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )

        super().__init__()
        self.self_attn = VIMultiheadAttention(
            d_model,
            nhead,
            batch_first=batch_first,
            bias=bias,
            **vikwargs,
        )
        self.multihead_attn = VIMultiheadAttention(
            d_model,
            nhead,
            batch_first=batch_first,
            bias=bias,
            **vikwargs,
        )
        # Feedforward model
        self._ff_block = VIResidualConnection(
            VILinear(d_model, dim_feedforward, bias=bias, **vikwargs),
            activation,
            VILinear(dim_feedforward, d_model, bias=bias, **vikwargs),
        )

        # layer norms are treated non-Bayesian
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)

        self._return_log_probs = return_log_probs

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
    ) -> VIReturn[Tensor]:
        """
        Pass the input through the decoder layer.

        See `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TranformerDecoderLayer.html>`__
        of :class:`nn.TranformerDecoderLayer` for details.

        This implementation also currently does not support the torch fastpath.
        """
        x = tgt
        # _ff_block already includes residual connection
        if self._return_log_probs and self.norm_first:
            (x1, _), lps1 = self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + x1
            (x2, _), lps2 = self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + x2
            x3, lps3 = self._ff_block(self.norm3(x))
            x = x3

            lps1, lps2, lps3 = cast(Tuple[Tensor, Tensor, Tensor], (lps1, lps2, lps3))
            log_probs = lps1 + lps2 + lps3
            return x, log_probs
        elif self._return_log_probs:
            (x1, _), lps1 = self._sa_block(
                x, tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = self.norm1(x + x1)
            (x2, _), lps2 = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
            )
            x = self.norm2(x + x2)
            x3, lps3 = self._ff_block(x)
            x = self.norm3(x3)

            lps1, lps2, lps3 = cast(Tuple[Tensor, Tensor, Tensor], (lps1, lps2, lps3))
            log_probs = lps1 + lps2 + lps3
            return x, log_probs
        elif self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)[0]
            x = (
                x
                + self._mha_block(
                    self.norm2(x),
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    tgt_is_causal,
                )[0]
            )
            x = self._ff_block(self.norm3(x))
            return x
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask)[0])
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )[0]
            )
            x = self.norm3(self._ff_block(x))
            return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> VIReturn[Tuple[Tensor, Optional[Tensor]]]:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return x

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> VIReturn[Tuple[Tensor, Optional[Tensor]]]:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return x


class VITransformerDecoder(VIModule):
    """
    TransformerDecoder is a stack of N decoder layers.

    Equivalent of :class:`nn.TransformerDecoder` with variational inference. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html>`__
    for usage.
    """

    def __init__(
        self,
        decoder_layer: "VITransformerDecoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
        return_log_probs: bool = True,
    ) -> None:
        super().__init__()
        self.layers = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm
        self.num_layers = num_layers
        self._return_log_probs = return_log_probs

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> VIReturn[Tensor]:
        """
        Pass the input through the decoder layers in turn.

        See `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TranformerDecoder.html>`__
        of :class:`nn.TransformerDecoder` for details.

        This implementation also currently does not support the torch fastpath.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        if self._return_log_probs:
            log_probs = torch.zeros(2, device=tgt.device)
            for mod in self.layers:
                output, lps = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_is_causal=tgt_is_causal,
                    memory_is_causal=memory_is_causal,
                )
                log_probs = log_probs + lps
            if self.norm is not None:
                output = self.norm(output)
            return output, log_probs
        else:
            for mod in self.layers:
                output = mod(
                    output,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_is_causal=tgt_is_causal,
                    memory_is_causal=memory_is_causal,
                )
            if self.norm is not None:
                output = self.norm(output)
            return output


class VITransformerEncoder(VIModule):
    """
    TransformerEncoder is a stack of N encoder layers.

    Equivalent of :class:`nn.TransformerEncoder` with variational inference. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html>`__
    for usage.
    """

    def __init__(
        self,
        encoder_layer: "VITransformerEncoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
        return_log_probs: bool = True,
    ) -> None:
        super().__init__()
        self.layers = ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm
        self.num_layers = num_layers
        self._return_log_probs = return_log_probs

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> VIReturn[Tensor]:
        """
        Pass the input through the encoder layers in turn.

        See `documentation <https://pytorch.org/docs/stable/generated/torch.nn.TranformerEncoder.html>`__
        of :class:`nn.TransformerEncoder` for details.

        This implementation also currently does not support the torch fastpath.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src

        seq_len = _get_seq_len(src, self.layers[0].self_attn.batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        if self._return_log_probs:
            log_probs = torch.zeros(2, device=src.device)
            for mod in self.layers:
                output, lps = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
                log_probs = log_probs + lps
            if self.norm is not None:
                output = self.norm(output)
            return output, log_probs
        else:
            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
            if self.norm is not None:
                output = self.norm(output)
            return output


class VITransformer(VIModule):
    """
    A Bayesian Transformer model.

    Equivalent of :class:`nn.Transformer` with variational inference. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__
    for usage.

    This does not support the ``dropout`` argument.
    In addition to all other arguments, this class accepts :class:`~.VIkwargs`.
    """

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 1,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 256,
        activation: Module = ReLU(),
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        variational_distribution: VariationalDistribution = MeanFieldNormalVarDist(),
        prior: Prior = MeanFieldNormalPrior(),
        rescale_prior: bool = True,
        kaiming_initialization: bool = True,
        prior_initialization: bool = False,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        vikwargs: VIkwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            rescale_prior=rescale_prior,
            kaiming_initialization=kaiming_initialization,
            prior_initialization=prior_initialization,
            return_log_probs=return_log_probs,
            device=device,
            dtype=dtype,
        )

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = VITransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **vikwargs,
            )
            encoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.encoder = VITransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm, return_log_probs
            )

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = VITransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation,
                layer_norm_eps,
                norm_first,
                batch_first,
                bias,
                **vikwargs,
            )
            decoder_norm = LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
            )
            self.decoder = VITransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm, return_log_probs
            )

        self.d_model = d_model
        self.nhead = nhead
        self._return_log_probs = return_log_probs

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        src_is_causal: Optional[bool] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> VIReturn[Tensor]:
        """
        Take in and process masked source/target sequences.

        See `documentation <https://pytorch.org/docs/stable/generated/torch.nn.Tranformer.html>`__
        of :class:`nn.Transformer` for details.

        This implementation also currently does not support the torch fastpath.
        """
        if self._return_log_probs:
            memory, encoder_log_probs = self.encoder(
                src,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=src_is_causal,
            )
            output, decoder_log_probs = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            log_probs = encoder_log_probs + decoder_log_probs
            return output, log_probs
        else:
            memory = self.encoder(
                src,
                mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=src_is_causal,
            )
            output = self.decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )
            return output
