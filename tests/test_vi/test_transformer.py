from typing import Any, Dict, Optional, Tuple, cast

import pytest
import torch
from pytest import mark, raises
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from torch_bayesian.vi import (
    VILinear,
    VIModule,
    VIMultiheadAttention,
    VITransformer,
    VITransformerDecoder,
    VITransformerDecoderLayer,
    VITransformerEncoder,
    VITransformerEncoderLayer,
)
from torch_bayesian.vi.priors import MeanFieldNormalPrior, Prior
from torch_bayesian.vi.variational_distributions import (
    MeanFieldNormalVarDist,
    VariationalDistribution,
)


class Filter(VIModule):
    """Wrapper to stop MultiheadAttention from returning None."""

    def __init__(self, module: VIMultiheadAttention) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Forward call."""
        out = self.module(*args, **kwargs)
        if isinstance(out, tuple) and out[1] is None:
            return out[0]
        else:
            return out


@pytest.mark.parametrize(
    "embed_dim,num_heads,variational_distribution,batch_size,src_len,tgt_len,use_attn_mask,use_key_mask,bias,add_bias_kv,add_zero_attn,kdim,vdim,batch_first,error",
    [
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            3,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            False,
            None,
            None,
            True,
            1,
        ),
        (
            32,
            4,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            True,
            False,
            True,
            False,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            True,
            True,
            True,
            False,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            True,
            True,
            False,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            False,
            False,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            True,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            False,
            True,
            False,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            True,
            None,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            False,
            None,
            None,
            False,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            False,
            30,
            None,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            False,
            None,
            16,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            True,
            False,
            False,
            10,
            36,
            True,
            None,
        ),
        (
            32,
            2,
            MeanFieldNormalVarDist(1e-20),
            3,
            5,
            7,
            False,
            False,
            False,
            False,
            False,
            16,
            16,
            True,
            None,
        ),
    ],
)
def test_multihead_attention(
    embed_dim: int,
    num_heads: int,
    variational_distribution: VariationalDistribution,
    batch_size: Optional[int],
    src_len: int,
    tgt_len: int,
    use_attn_mask: bool,
    use_key_mask: bool,
    bias: bool,
    add_bias_kv: bool,
    add_zero_attn: bool,
    kdim: Optional[int],
    vdim: Optional[int],
    batch_first: bool,
    error: Optional[int],
    device: torch.device,
) -> None:
    """Test VIMultiheadAttention."""
    samples = 100
    primary_param = variational_distribution.variational_parameters[0]

    if error == 1:
        with raises(AssertionError, match="embed_dim must be divisible by num_heads"):
            _ = VIMultiheadAttention(
                embed_dim,
                num_heads,
                bias=bias,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
                batch_first=batch_first,
                device=device,
            )
        return

    module = Filter(
        VIMultiheadAttention(
            embed_dim,
            num_heads,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            variational_distribution=variational_distribution,
            device=device,
        )
    )

    random_variable_shapes: Dict[str, Optional[Tuple[int, ...]]] = dict(
        in_proj_weight=None,
        q_proj_weight=None,
        k_proj_weight=None,
        v_proj_weight=None,
        out_proj_weight=None,
        in_proj_bias=None,
        out_proj_bias=None,
        bias_k=None,
        bias_v=None,
    )
    if kdim is None and vdim is None:
        use_separate_proj_weight = False
        assert module.module._qkv_same_embed_dim
        random_variable_shapes["in_proj_weight"] = (3 * embed_dim, embed_dim)
    else:
        use_separate_proj_weight = True
        assert not module.module._qkv_same_embed_dim
        random_variable_shapes["q_proj_weight"] = (embed_dim, embed_dim)
        random_variable_shapes["k_proj_weight"] = (embed_dim, kdim or embed_dim)
        random_variable_shapes["v_proj_weight"] = (embed_dim, vdim or embed_dim)
    random_variable_shapes["out_proj_weight"] = (embed_dim, embed_dim)
    if bias:
        random_variable_shapes["in_proj_bias"] = (3 * embed_dim,)
        random_variable_shapes["out_proj_bias"] = (embed_dim,)
    if add_bias_kv:
        random_variable_shapes["bias_k"] = (1, 1, embed_dim)
        random_variable_shapes["bias_v"] = (1, 1, embed_dim)

    assert module.module.embed_dim == embed_dim
    assert module.module.kdim == (kdim or embed_dim)
    assert module.module.vdim == (vdim or embed_dim)
    assert module.module.num_heads == num_heads
    assert module.module.bias == bias
    assert module.module.batch_first == batch_first
    module_random_vars = cast(Tuple[str, ...], module.module.random_variables)
    assert len(module_random_vars) == len(random_variable_shapes.keys())
    for v1, v2 in zip(module_random_vars, random_variable_shapes.keys()):
        assert v1 == v2

    param_dict = dict(module.module.named_parameters())
    for var, shape in random_variable_shapes.items():
        if module.module.variational_distribution[var] is None:
            continue
        for param in variational_distribution.variational_parameters:
            name = module.module.variational_parameter_name(var, param)
            assert name in param_dict
            assert param_dict[name].shape == shape
            assert param_dict[name].device == device

    if batch_size is not None:
        src_shape: Tuple[int, ...] = (batch_size, src_len, embed_dim)
        tgt_shape: Tuple[int, ...] = (batch_size, tgt_len, kdim or embed_dim)
        ext_shape: Tuple[int, ...] = (batch_size, tgt_len, vdim or embed_dim)
    else:
        src_shape = (src_len, embed_dim)
        tgt_shape = (tgt_len, kdim or embed_dim)
        ext_shape = (tgt_len, vdim or embed_dim)

    src1 = torch.rand(src_shape, device=device)
    tgt1 = torch.rand(tgt_shape, device=device)
    extr = torch.rand(ext_shape, device=device)

    weight_dict = dict(
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        out_proj_weight=None,
        out_proj_bias=None,
    )
    for var in random_variable_shapes:
        if module.module.variational_distribution[var] is None:
            continue
        weight_dict[var] = getattr(
            module.module, module.module.variational_parameter_name(var, primary_param)
        ).clone()

    for q, k, v in [(src1, src1, src1), (src1, tgt1, tgt1), (src1, tgt1, extr)]:
        if torch.equal(k, v) and ((kdim is not None) or (vdim is not None)):
            continue

        if use_attn_mask:
            attn_mask = torch.tril(
                torch.full((q.shape[-2], k.shape[-2]), float("-inf"), device=device), -1
            )
        else:
            attn_mask = None

        if use_key_mask:
            if batch_size is not None:
                shape = (batch_size, k.shape[-2])
            else:
                shape = (k.shape[-2],)
            x = torch.rand(shape, device=device)
            key_mask = torch.zeros_like(x, device=device).where(x > 0.1, float("-inf"))
        else:
            key_mask = None

        ref_args = dict(
            query=q.transpose(0, 1),
            key=k.transpose(0, 1),
            value=v.transpose(0, 1),
            embed_dim_to_check=embed_dim,
            num_heads=num_heads,
            add_zero_attn=add_zero_attn,
            dropout_p=0.0,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            need_weights=True,
            key_padding_mask=key_mask,
            average_attn_weights=False,
            **weight_dict,
        )

        ref, ref_weights = F.multi_head_attention_forward(**ref_args)

        if not batch_first:
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
        else:
            ref = ref.transpose(0, 1)

        model_return = module(
            q,
            k,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_mask,
            average_attn_weights=False,
            samples=samples,
        )

        out, weights = model_return

        weights = weights.mean(dim=0)
        weights = torch.nan_to_num(weights)
        ref_weights = torch.nan_to_num(ref_weights)
        assert ref_weights.shape == weights.shape
        assert torch.allclose(ref_weights, weights, atol=1e-6)
        assert weights.device == device

        out = out.mean(dim=0)
        out = torch.nan_to_num(out)
        ref = torch.nan_to_num(ref)
        out.sum().backward()
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-6)
        assert out.device == device


def test_decoder_layer(device: torch.device) -> None:
    """Test VITransformerDecoderLayer."""
    d_model = 8
    nhead = 2

    module1 = VITransformerDecoderLayer(d_model, nhead, device=device)
    assert not module1.norm_first
    assert module1.self_attn.embed_dim == d_model
    assert module1.self_attn.num_heads == nhead
    assert module1.self_attn.bias
    assert module1.self_attn.batch_first
    assert module1.multihead_attn.embed_dim == d_model
    assert module1.multihead_attn.num_heads == nhead
    assert module1.multihead_attn.bias
    assert module1.multihead_attn.batch_first
    assert module1._ff_block[0].in_features == d_model
    assert module1._ff_block[0].out_features == 512
    assert isinstance(module1._ff_block[1], nn.ReLU)
    assert module1._ff_block[2].in_features == 512
    assert module1._ff_block[2].out_features == d_model
    assert module1.norm1.eps == 1e-5
    assert module1.norm2.eps == 1e-5
    assert module1.norm3.eps == 1e-5
    assert module1.norm1.bias is not None
    assert module1.norm2.bias is not None
    assert module1.norm3.bias is not None

    module2 = VITransformerDecoderLayer(
        d_model,
        nhead,
        dim_feedforward=128,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        activation=nn.GELU(),
        layer_norm_eps=1e-3,
        norm_first=True,
        bias=False,
        batch_first=False,
        device=device,
    )
    assert module2.norm_first
    assert module2.self_attn.embed_dim == d_model
    assert module2.self_attn.num_heads == nhead
    assert not module2.self_attn.bias
    assert not module2.self_attn.batch_first
    assert module2.multihead_attn.embed_dim == d_model
    assert module2.multihead_attn.num_heads == nhead
    assert not module2.multihead_attn.bias
    assert not module2.multihead_attn.batch_first
    assert module2._ff_block[0].in_features == d_model
    assert module2._ff_block[0].out_features == 128
    assert isinstance(module2._ff_block[1], nn.GELU)
    assert module2._ff_block[2].in_features == 128
    assert module2._ff_block[2].out_features == d_model
    assert module2.norm1.eps == 1e-3
    assert module2.norm2.eps == 1e-3
    assert module2.norm3.eps == 1e-3
    assert module2.norm1.bias is None
    assert module2.norm2.bias is None
    assert module2.norm3.bias is None

    module3 = VITransformerDecoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        norm_first=False,
        bias=False,
        device=device,
    )

    tgt = torch.rand((7, 4, d_model), device=device)
    mem = torch.rand((7, 4, d_model), device=device)

    sa_ref, sa_rweight = module2.self_attn(tgt, tgt, tgt)
    sa_rlp = module2.gather_log_probs()
    sa_out, sa_weights = module2._sa_block(tgt)
    sa_lps = module2.gather_log_probs()
    assert sa_out.shape == sa_ref.shape
    assert torch.allclose(sa_out, sa_ref)
    assert torch.allclose(sa_lps, sa_rlp)
    assert sa_out.device == device
    assert sa_ref.device == device

    mha_out, mha_weights = module2._mha_block(tgt, mem)
    mha_lps = module2.gather_log_probs()
    mha_ref, mha_rweight = module2.multihead_attn(tgt, mem, mem)
    mha_rlp = module2.gather_log_probs()
    assert mha_out.shape == mha_ref.shape
    assert torch.allclose(mha_out, mha_ref)
    assert torch.allclose(mha_lps, mha_rlp)
    assert mha_out.device == device
    assert mha_ref.device == device

    # check norm_first=True
    module2.return_log_probs = False
    out1 = module2(tgt, mem)
    out1.sum().backward()
    ref1 = tgt + module2._sa_block(module2.norm1(tgt))[0]
    ref1 = ref1 + module2._mha_block(module2.norm2(ref1), mem)[0]
    ref1 = module2._ff_block(module2.norm3(ref1))
    assert torch.allclose(out1, ref1, atol=1e-6)

    module2.return_log_probs = True
    out2 = module2(tgt, mem)
    out2.sum().backward()
    assert torch.allclose(out1, out2, atol=1e-6)

    # check norm_first=False
    module3.return_log_probs = False
    out3 = module3(tgt, mem)
    out3.sum().backward()
    ref2 = module3.norm1(tgt + module3._sa_block(tgt)[0])
    ref2 = module3.norm2(ref2 + module3._mha_block(ref2, mem)[0])
    ref2 = module3.norm3(module3._ff_block(ref2))
    assert torch.allclose(out3, ref2, atol=1e-6)

    module3.return_log_probs = True
    out4 = module3(tgt, mem)
    out4.sum().backward()
    assert torch.allclose(out3, out4, atol=1e-6)


def test_encoder_layer(device: torch.device) -> None:
    """Test VITransformerEncoderLayer."""
    d_model = 8
    nhead = 2

    module1 = VITransformerEncoderLayer(d_model, nhead, device=device)
    assert not module1.norm_first
    assert module1.self_attn.embed_dim == d_model
    assert module1.self_attn.num_heads == nhead
    assert module1.self_attn.bias
    assert module1.self_attn.batch_first
    assert module1._ff_block[0].in_features == d_model
    assert module1._ff_block[0].out_features == 512
    assert isinstance(module1._ff_block[1], nn.ReLU)
    assert module1._ff_block[2].in_features == 512
    assert module1._ff_block[2].out_features == d_model
    assert module1.norm1.eps == 1e-5
    assert module1.norm2.eps == 1e-5
    assert module1.norm1.bias is not None
    assert module1.norm2.bias is not None

    module2 = VITransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=128,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        activation=nn.GELU(),
        layer_norm_eps=1e-3,
        norm_first=True,
        bias=False,
        batch_first=False,
        device=device,
    )
    assert module2.norm_first
    assert module2.self_attn.embed_dim == d_model
    assert module2.self_attn.num_heads == nhead
    assert not module2.self_attn.bias
    assert not module2.self_attn.batch_first
    assert module2._ff_block[0].in_features == d_model
    assert module2._ff_block[0].out_features == 128
    assert isinstance(module2._ff_block[1], nn.GELU)
    assert module2._ff_block[2].in_features == 128
    assert module2._ff_block[2].out_features == d_model
    assert module2.norm1.eps == 1e-3
    assert module2.norm2.eps == 1e-3
    assert module2.norm1.bias is None
    assert module2.norm2.bias is None

    module3 = VITransformerEncoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        norm_first=False,
        bias=False,
        device=device,
    )

    src = torch.rand((7, 4, d_model), device=device)

    sa_out, sa_weights = module2._sa_block(src)
    sa_lps = module2.gather_log_probs()
    sa_ref, sa_rweight = module2.self_attn(src, src, src)
    sa_rlp = module2.gather_log_probs()
    assert sa_out.shape == sa_ref.shape
    assert torch.allclose(sa_out, sa_ref)
    assert torch.allclose(sa_lps, sa_rlp)
    assert sa_out.device == device
    assert sa_ref.device == device

    # check norm_first=True
    module2.return_log_probs = False
    out1 = module2(src)
    out1.sum().backward()
    ref1 = src + module2._sa_block(module2.norm1(src))[0]
    ref1 = module2._ff_block(module2.norm2(ref1))
    assert torch.allclose(out1, ref1)

    module2.return_log_probs = True
    out2 = module2(src)
    out2.sum().backward()
    assert torch.allclose(out1, out2, atol=2e-7)

    # check norm_first=False
    module3.return_log_probs = False
    out3 = module3(src)
    out3.sum().backward()
    ref2 = module3.norm1(src + module3._sa_block(src)[0])
    ref2 = module3.norm2(module3._ff_block(ref2))
    assert torch.allclose(out3, ref2, atol=2e-7)

    module3.return_log_probs = True
    out4 = module3(src)
    out4.sum().backward()
    assert torch.allclose(out3, out4, atol=2e-7)


def test_decoder(device: torch.device) -> None:
    """Test VITransformerDecoder."""
    d_model = 8
    nhead = 2
    num_layers = 3

    layer = VITransformerDecoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        device=device,
    )
    module1 = VITransformerDecoder(layer, num_layers)

    assert len(module1.layers) == num_layers
    assert module1.norm is None
    assert module1.num_layers == num_layers

    for lay in module1.layers:
        assert isinstance(lay, VITransformerDecoderLayer)
        assert lay.self_attn.embed_dim == d_model
        assert lay.self_attn.num_heads == nhead
        assert lay is not layer
    assert module1.layers[0] is not module1.layers[1]

    tgt = torch.rand((9, 5, d_model), device=device)
    memory = torch.rand((9, 5, d_model), device=device)

    module1.return_log_probs = False
    out1 = module1(tgt, memory)
    out1.sum().backward()
    ref = tgt
    for mod in module1.layers:
        ref = mod(ref, memory)

    assert out1.shape == (10, 9, 5, d_model)
    assert out1.shape[1:] == ref.shape
    assert torch.allclose(out1[0], ref, atol=1e-6)
    assert out1.device == device

    module1.return_log_probs = True
    out2 = module1(tgt, memory)
    out2.sum().backward()
    assert out1.shape == out2.shape
    assert torch.allclose(out1, out2, atol=1e-6)
    assert out1.device == device

    module2 = VITransformerDecoder(
        layer, num_layers, norm=nn.LayerNorm(d_model, device=device)
    )
    assert isinstance(module2.norm, nn.LayerNorm)

    module2.return_log_probs = False
    out3 = module2(tgt, memory)
    out3.sum().backward()
    ref2 = tgt
    for mod in module2.layers:
        ref2 = mod(ref2, memory)
    ref2 = module2.norm(ref2)

    assert out3.shape == (10, 9, 5, d_model)
    assert out3.shape[1:] == ref2.shape
    assert torch.allclose(out3[0], ref2, atol=1e-6)
    assert out3.device == device

    module2.return_log_probs = True
    out4 = module2(tgt, memory)
    out4.sum().backward()
    assert out3.shape == out4.shape
    assert torch.allclose(out3, out4, atol=1e-6)
    assert out4.device == device


def test_encoder(device: torch.device) -> None:
    """Test VITransformerEncoder."""
    d_model = 8
    nhead = 2
    num_layers = 3

    layer = VITransformerEncoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        device=device,
    )
    module1 = VITransformerEncoder(layer, num_layers)

    assert len(module1.layers) == num_layers
    assert module1.norm is None
    assert module1.num_layers == num_layers

    for lay in module1.layers:
        assert isinstance(lay, VITransformerEncoderLayer)
        assert lay.self_attn.embed_dim == d_model
        assert lay.self_attn.num_heads == nhead
        assert lay is not layer
    assert module1.layers[0] is not module1.layers[1]

    src = torch.rand((9, 5, d_model), device=device)

    module1.return_log_probs = False
    out1 = module1(src)
    out1.sum().backward()
    ref = src
    for mod in module1.layers:
        ref = mod(ref)

    assert out1.shape == (10, 9, 5, d_model)
    assert out1.shape[1:] == ref.shape
    assert torch.allclose(out1[0], ref, atol=1e-6)
    assert out1.device == device

    module1.return_log_probs = True
    out2 = module1(src)
    out2.sum().backward()
    assert out1.shape == out2.shape
    assert torch.allclose(out1, out2, atol=2e-7)
    assert out2.device == device

    module2 = VITransformerEncoder(
        layer, num_layers, norm=nn.LayerNorm(d_model, device=device)
    )
    assert isinstance(module2.norm, nn.LayerNorm)

    module2.return_log_probs = False
    out3 = module2(src)
    out3.sum().backward()
    ref2 = src
    for mod in module2.layers:
        ref2 = mod(ref2)
    ref2 = module2.norm(ref2)

    assert out3.shape == (10, 9, 5, d_model)
    assert out3.shape[1:] == ref2.shape
    assert torch.allclose(out3[0], ref2, atol=1e-6)
    assert out3.device == device

    module2.return_log_probs = True
    out4 = module2(src)
    out4.sum().backward()
    assert out3.shape == out4.shape
    assert torch.allclose(out3, out4, atol=1e-6)
    assert out4.device == device


@mark.parametrize(
    "d_model,nhead,num_encoder_layers,num_decoder_layers,dim_feedforward,activation,"
    "custom_coders,layer_norm_eps,batch_first,norm_first,bias,variational_distribution,"
    "prior,prior_initialization,rescale_prior,return_log_probs,dtype",
    [
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            27,
            3,
            2,
            3,
            56,
            nn.GELU(),
            False,
            1e-7,
            True,
            True,
            False,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            True,
            False,
            False,
            torch.float16,
        ),
        (
            27,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            27,
            3,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            4,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            2,
            3,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            56,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.GELU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-7,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            True,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            True,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            False,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            True,
            True,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            False,
            True,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            False,
            None,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            False,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            True,
            torch.float16,
        ),
        (
            32,
            1,
            1,
            1,
            256,
            nn.ReLU(),
            True,
            1e-5,
            False,
            False,
            True,
            MeanFieldNormalVarDist(initial_std=1e-20),
            MeanFieldNormalPrior(),
            False,
            True,
            False,
            None,
        ),
    ],
)
def test_transformer(
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    activation: nn.Module,
    custom_coders: bool,
    layer_norm_eps: float,
    batch_first: bool,
    norm_first: bool,
    bias: bool,
    variational_distribution: VariationalDistribution,
    prior: Prior,
    prior_initialization: bool,
    rescale_prior: bool,
    return_log_probs: bool,
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    num_samples: int = 7,
    batch_size: int = 3,
    src_length: int = 5,
    tgt_length: int = 4,
) -> None:
    """Test VITransformer."""
    if custom_coders:

        class ArgWrapper(nn.Module):
            def __init__(self, coder: nn.Module) -> None:
                super().__init__()
                self.coder = coder

            def forward(self, input_: Tensor, *args: Any, **kwargs: Any) -> Tensor:
                return self.coder(input_)

        encoder = ArgWrapper(nn.Linear(d_model, d_model, device=device))
        decoder = VITransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            activation,
            layer_norm_eps,
            norm_first,
            batch_first,
            bias,
            variational_distribution,
            prior,
            rescale_prior,
            True,
            prior_initialization,
            return_log_probs,
            device,
            dtype,
        )
        module = VITransformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            activation,
            encoder,
            decoder,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            variational_distribution,
            prior,
            rescale_prior,
            True,
            prior_initialization,
            return_log_probs,
            device,
            dtype,
        )
        assert module.encoder is encoder
        assert module.decoder is decoder
    else:
        module = VITransformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            activation,
            None,
            None,
            layer_norm_eps,
            batch_first,
            norm_first,
            bias,
            variational_distribution,
            prior,
            rescale_prior,
            True,
            prior_initialization,
            return_log_probs,
            device,
            dtype,
        )

        # Set default device and dtype
        dtype = dtype or torch.float32

        assert len(module.encoder.layers) == num_encoder_layers
        assert len(module.decoder.layers) == num_decoder_layers

        for name, param in module.named_parameters():
            assert param.device == device
            assert param.dtype == dtype

        for layer in module.encoder.layers:
            assert layer.self_attn.embed_dim == d_model
            assert layer.self_attn.num_heads == nhead
            assert layer.self_attn.bias == bias
            assert layer.self_attn.batch_first == batch_first

        for layer in module.decoder.layers:
            assert layer.self_attn.embed_dim == d_model
            assert layer.self_attn.num_heads == nhead
            assert layer.self_attn.bias == bias
            assert layer.self_attn.batch_first == batch_first
            assert layer.multihead_attn.embed_dim == d_model
            assert layer.multihead_attn.num_heads == nhead
            assert layer.multihead_attn.bias == bias
            assert layer.multihead_attn.batch_first == batch_first

        for layer in module.modules():
            if isinstance(layer, VIModule):
                if not layer._return_log_probs == return_log_probs:
                    print(f"{layer.__class__.__name__}")
                assert (
                    layer._return_log_probs == return_log_probs
                ), f"{layer.__class__.__name__}"
                if layer.random_variables is not None:
                    # Check vardist and prior propagate to all VIBaseLayers
                    for var_dist, prior in zip(
                        layer.variational_distribution.values(), layer.prior.values()
                    ):
                        if var_dist is None:
                            assert prior is None
                            continue
                        assert isinstance(var_dist, type(variational_distribution))
                        assert isinstance(prior, type(prior))
                    # Check bias propagates to all VIBaseLayers
                    if isinstance(layer, VILinear):
                        bias_mean_name = layer.variational_parameter_name(
                            "bias", "mean"
                        )
                        bias_log_std_name = layer.variational_parameter_name(
                            "bias", "log_std"
                        )
                        assert hasattr(layer, bias_mean_name) == bias
                        assert hasattr(layer, bias_log_std_name) == bias
                    else:
                        assert layer.bias == bias

            if isinstance(layer, nn.LayerNorm):
                assert layer.eps == layer_norm_eps
                assert (layer.bias is not None) == bias

    sample_src = torch.rand(
        [batch_size, src_length, d_model], dtype=dtype, device=device
    )
    sample_tgt = torch.rand(
        [batch_size, tgt_length, d_model], dtype=dtype, device=device
    )
    output_shape = (num_samples, batch_size, tgt_length, d_model)

    if not batch_first:
        sample_src = sample_src.transpose(0, 1)
        sample_tgt = sample_tgt.transpose(0, 1)
        output_shape = (num_samples, tgt_length, batch_size, d_model)

    sample_output = module(sample_src, sample_tgt, samples=num_samples)

    if return_log_probs:
        log_probs = sample_output.log_probs
        sample_output += (
            log_probs.sum()
        )  # this makes the backward below also track logprobs
        assert log_probs.shape == (num_samples, 2)
        assert log_probs.device == device

    sample_output.sum().backward()
    assert sample_output.shape == output_shape
    assert sample_output.device == device
