"""LayerNorm kernel written in Triton.

NOT VALIDATED ON GPU YET. Requires a CUDA-capable GPU and Triton.

Semantics match ``torch.nn.LayerNorm`` with affine weight/bias and a fixed
``eps`` (the reference uses ``eps=1e-5``):

    mean  = sum(x) / D
    var   = sum(x^2) / D - mean^2   (clamped to >= 0)
    y     = (x - mean) * rsqrt(var + eps) * weight + bias

Statistics are accumulated in fp32 regardless of input dtype; the output is
cast back to the input dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _next_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@triton.jit
def _layernorm_fwd(
    X, W, B, Y,
    eps,
    D: tl.constexpr,
    stride_xb, stride_xs, stride_xd,
    stride_wd, stride_bd,
    stride_yb, stride_ys, stride_yd,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    b = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x_ptrs = X + b * stride_xb + row * stride_xs + offs_d * stride_xd
    x = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    var = tl.sum(x * x, axis=0) / D - mean * mean
    rstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)

    w = tl.load(W + offs_d * stride_wd, mask=mask_d, other=0.0).to(tl.float32)
    beta = tl.load(B + offs_d * stride_bd, mask=mask_d, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + beta
    y = y.to(Y.dtype.element_ty)

    y_ptrs = Y + b * stride_yb + row * stride_ys + offs_d * stride_yd
    tl.store(y_ptrs, y, mask=mask_d)


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply LayerNorm to a ``[B, S, D]`` tensor with affine parameters."""
    if x.dim() != 3:
        raise ValueError(f"expected [B, S, D], got {tuple(x.shape)}")
    batch, seq_len, dim = x.shape
    if dim <= 0 or dim > 4096:
        raise ValueError(f"unsupported D={dim} for the LayerNorm kernel")
    if weight.numel() != dim or bias.numel() != dim:
        raise ValueError("weight/bias must have D elements")

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    y = torch.empty_like(x)

    block_d = _next_pow2(dim)
    grid = (seq_len, batch)
    _layernorm_fwd[grid](
        x, weight, bias, y,
        float(eps),
        D=dim,
        stride_xb=x.stride(0), stride_xs=x.stride(1), stride_xd=x.stride(2),
        stride_wd=weight.stride(0), stride_bd=bias.stride(0),
        stride_yb=y.stride(0), stride_ys=y.stride(1), stride_yd=y.stride(2),
        BLOCK_D=block_d,
    )
    return y
