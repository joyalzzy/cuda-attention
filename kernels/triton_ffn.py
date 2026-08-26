"""Experimental fused FFN path written in Triton.

NOT VALIDATED ON GPU YET. Requires a CUDA-capable GPU and Triton. This path is
**opt-in** (``TORCH_TRANSFORMER_FFN_KERNEL=1``) because it is the least mature
part of the kernel set and the GEMM tiling recomputes/rewrites intermediate
tiles, trading compute for fewer kernel launches.

It replaces, per Transformer block:

    h1 = gelu(ffn_in(layernorm(x)))         # exact GELU, erf-based
    y  = ffn_out(h1) + x                    # + padding-row zeroing

with two fused kernels:
1. ``_fused_ln_gemm1_gelu``: LayerNorm + GEMM1 + bias + exact GELU -> ``h1``.
2. ``_fused_gemm2_residual``: GEMM2 + bias + residual add + zeroing -> ``y``.

Semantics match the reference: LayerNorm eps 1e-5, affine weight/bias, fp32
statistics, exact GELU, fp32 accumulation for fp16/bf16, and ``input_precision
= "ieee"`` for fp32 dots.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

_GELU_COEF = 0.7071067811865476  # 1 / sqrt(2)


@triton.jit
def _fused_ln_gemm1_gelu(
    X, Gamma, Beta, W1, B1, H1,
    eps,
    S, F,
    stride_xb, stride_xs, stride_xd,
    stride_gd, stride_bd,
    stride_w1f, stride_w1d,
    stride_b1f,
    stride_h1b, stride_h1s, stride_h1f,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_F: tl.constexpr,
    PRECISE: tl.constexpr,
):
    start_m = tl.program_id(0)
    b = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < S

    # Pass 1: LayerNorm statistics over D.
    sum_x = tl.zeros((BLOCK_M,), tl.float32)
    sum_x2 = tl.zeros((BLOCK_M,), tl.float32)
    for start_d in range(0, tl.cdiv(D, BLOCK_D)):
        offs_d = start_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        x = tl.load(
            X + b * stride_xb + offs_m[:, None] * stride_xs + offs_d[None, :] * stride_xd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        sum_x += tl.sum(x, axis=1)
        sum_x2 += tl.sum(x * x, axis=1)
    mean = sum_x / D
    var = sum_x2 / D - mean * mean
    rstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)

    # Pass 2: affine LayerNorm + GEMM1 + bias + exact GELU -> h1 tile.
    # Grid dimension 2 indexes the FFN output tile; every F tile is computed.
    start_f = tl.program_id(2)
    offs_f = start_f * BLOCK_F + tl.arange(0, BLOCK_F)
    acc = tl.zeros((BLOCK_M, BLOCK_F), tl.float32)
    b1 = tl.load(B1 + offs_f * stride_b1f, mask=offs_f < F, other=0.0).to(tl.float32)
    acc += b1[None, :]
    for start_d in range(0, tl.cdiv(D, BLOCK_D)):
        offs_d = start_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        x = tl.load(
            X + b * stride_xb + offs_m[:, None] * stride_xs + offs_d[None, :] * stride_xd,
            mask=row_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gamma = tl.load(
            Gamma + offs_d * stride_gd,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(
            Beta + offs_d * stride_bd,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        xn = (x - mean[:, None]) * rstd[:, None]
        xn = xn * gamma[None, :] + beta[None, :]
        xn = xn.to(X.dtype.element_ty)
        w1 = tl.load(
            W1 + offs_f[:, None] * stride_w1f + offs_d[None, :] * stride_w1d,
            mask=(offs_f[:, None] < F) & d_mask[None, :],
            other=0.0,
        ).to(X.dtype.element_ty)
        w1_t = tl.trans(w1)
        if PRECISE:
            acc = tl.dot(xn, w1_t, acc=acc, input_precision="ieee")
        else:
            acc = tl.dot(xn, w1_t, acc=acc)

    h = 0.5 * acc * (1.0 + tl.math.erf(acc * _GELU_COEF))
    h = h.to(H1.dtype.element_ty)
    tl.store(
        H1 + b * stride_h1b + offs_m[:, None] * stride_h1s + offs_f[None, :] * stride_h1f,
        h,
        mask=row_mask[:, None] & (offs_f[None, :] < F),
    )


@triton.jit
def _fused_gemm2_residual(
    H1, W2, B2, X, Valid, Y,
    S, D,
    stride_h1b, stride_h1s, stride_h1f,
    stride_w2d, stride_w2f,
    stride_b2d,
    stride_xb, stride_xs, stride_xd,
    stride_vb, stride_vs,
    stride_yb, stride_ys, stride_yd,
    F: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_F: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_MASK: tl.constexpr,
    PRECISE: tl.constexpr,
):
    start_d = tl.program_id(0)
    start_m = tl.program_id(1)
    b = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = start_d * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = offs_m < S
    d_mask = offs_d < D

    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    b2 = tl.load(B2 + offs_d * stride_b2d, mask=d_mask, other=0.0).to(tl.float32)
    acc += b2[None, :]

    for start_f in range(0, tl.cdiv(F, BLOCK_F)):
        offs_f = start_f * BLOCK_F + tl.arange(0, BLOCK_F)
        f_mask = offs_f < F
        h1 = tl.load(
            H1 + b * stride_h1b + offs_m[:, None] * stride_h1s + offs_f[None, :] * stride_h1f,
            mask=row_mask[:, None] & f_mask[None, :],
            other=0.0,
        )
        w2 = tl.load(
            W2 + offs_d[:, None] * stride_w2d + offs_f[None, :] * stride_w2f,
            mask=d_mask[:, None] & f_mask[None, :],
            other=0.0,
        )
        w2_t = tl.trans(w2)
        if PRECISE:
            acc = tl.dot(h1, w2_t, acc=acc, input_precision="ieee")
        else:
            acc = tl.dot(h1, w2_t, acc=acc)

    x = tl.load(
        X + b * stride_xb + offs_m[:, None] * stride_xs + offs_d[None, :] * stride_xd,
        mask=row_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    y = acc + x
    if HAS_MASK:
        valid = tl.load(
            Valid + b * stride_vb + offs_m * stride_vs,
            mask=row_mask,
            other=0,
        )
        y = tl.where(valid[:, None] > 0, y, 0.0)
    y = y.to(Y.dtype.element_ty)
    tl.store(
        Y + b * stride_yb + offs_m[:, None] * stride_ys + offs_d[None, :] * stride_yd,
        y,
        mask=row_mask[:, None] & d_mask[None, :],
    )


def fused_ffn(
    x: torch.Tensor,
    norm: torch.nn.Module,
    ffn_in: torch.nn.Module,
    ffn_out: torch.nn.Module,
    valid_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``x + ffn_out(gelu(ffn_in(layernorm(x))))`` with padding zeroing."""
    if x.dim() != 3:
        raise ValueError(f"expected [B, S, D], got {tuple(x.shape)}")
    batch, seq_len, dim = x.shape
    ffn_dim = ffn_in.out_features
    if ffn_dim != ffn_out.in_features:
        raise ValueError("ffn_in/ffn_out dimensions do not line up")
    if dim % 16 != 0 or ffn_dim % 16 != 0:
        raise ValueError("fused FFN requires D and F divisible by 16")
    if dim > 2048 or ffn_dim > 8192:
        raise ValueError("fused FFN shapes exceed kernel caps")

    x = x.contiguous()
    w1 = ffn_in.weight.contiguous()
    b1 = ffn_in.bias.contiguous()
    w2 = ffn_out.weight.contiguous()
    b2 = ffn_out.bias.contiguous()
    gamma = norm.weight.contiguous()
    beta = norm.bias.contiguous()
    h1 = torch.empty((batch, seq_len, ffn_dim), dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)

    has_mask = valid_token_mask is not None
    if has_mask:
        valid = valid_token_mask.to(torch.int8).contiguous()
        stride_vb, stride_vs = valid.stride(0), valid.stride(1)
    else:
        valid = torch.empty(0, dtype=torch.int8, device=x.device)
        stride_vb, stride_vs = 0, 0

    precise = x.dtype == torch.float32
    block_m = 64
    block_d = 64
    block_f = 64

    # Kernel A: LayerNorm + GEMM1 + GELU -> h1.
    grid_a = (triton.cdiv(seq_len, block_m), batch, triton.cdiv(ffn_dim, block_f))
    _fused_ln_gemm1_gelu[grid_a](
        x, gamma, beta, w1, b1, h1,
        float(norm.eps), seq_len, ffn_dim,
        x.stride(0), x.stride(1), x.stride(2),
        gamma.stride(0), beta.stride(0),
        w1.stride(0), w1.stride(1),
        b1.stride(0),
        h1.stride(0), h1.stride(1), h1.stride(2),
        D=dim, BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_F=block_f,
        PRECISE=precise,
    )

    # Kernel B: GEMM2 + bias + residual + zeroing -> y.
    grid_b = (triton.cdiv(dim, block_d), triton.cdiv(seq_len, block_m), batch)
    _fused_gemm2_residual[grid_b](
        h1, w2, b2, x, valid, y,
        seq_len, dim,
        h1.stride(0), h1.stride(1), h1.stride(2),
        w2.stride(0), w2.stride(1),
        b2.stride(0),
        x.stride(0), x.stride(1), x.stride(2),
        stride_vb, stride_vs,
        y.stride(0), y.stride(1), y.stride(2),
        F=ffn_dim, BLOCK_M=block_m, BLOCK_F=block_f, BLOCK_D=block_d,
        HAS_MASK=has_mask, PRECISE=precise,
    )
    return y
