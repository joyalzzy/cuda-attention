"""Kernel availability checks, shape gating, and dispatch policy.

Environment variables:

- ``TORCH_TRANSFORMER_DISABLE_KERNELS=1`` disables all Triton kernels
  (always falls back to SDPA / reference paths).
- ``TORCH_TRANSFORMER_FFN_KERNEL=1`` enables the experimental fused FFN path
  (off by default).

Kernels only ever run on CUDA devices. Any shape outside the supported window,
any import/runtime failure, or a disabled flag routes execution to the
framework fallback path.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

KERNELS_DISABLED = os.environ.get("TORCH_TRANSFORMER_DISABLE_KERNELS", "0") == "1"
FFN_ENABLED = os.environ.get("TORCH_TRANSFORMER_FFN_KERNEL", "0") == "1"


def triton_available() -> bool:
    """True when Triton is importable and a CUDA device is present."""
    if KERNELS_DISABLED:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def attention_kernel_supported(seq_len: int, head_dim: int) -> bool:
    """Flash attention kernel shape window."""
    return 16 <= seq_len and 8 <= head_dim <= 256


def layernorm_kernel_supported(dim: int) -> bool:
    """LayerNorm kernel shape window (any D up to a register/block cap)."""
    return 16 <= dim <= 4096


def ffn_kernel_supported(dim: int, ffn_dim: int) -> bool:
    """Experimental fused FFN shape window (needs 16-aligned tensor-core dims)."""
    if not FFN_ENABLED:
        return False
    if dim % 16 != 0 or ffn_dim % 16 != 0:
        return False
    return 16 <= dim <= 2048 and 16 <= ffn_dim <= 8192


def choose_attention(
    device: torch.device,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> str:
    """Return 'triton', 'sdpa', or 'reference' for one attention call."""
    if (
        triton_available()
        and device.type == "cuda"
        and attention_kernel_supported(seq_len, head_dim)
    ):
        return "triton"
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        if device.type != "cuda" and dtype in (torch.float16, torch.bfloat16):
            return "reference"
        return "sdpa"
    return "reference"


def choose_layernorm(device: torch.device, dim: int) -> bool:
    """True when the LayerNorm kernel should be used for a ``[.., D]`` tensor."""
    return (
        triton_available()
        and device.type == "cuda"
        and layernorm_kernel_supported(dim)
    )


def choose_ffn(device: torch.device, dim: int, ffn_dim: int) -> bool:
    """True when the experimental fused FFN path should be used."""
    return (
        triton_available()
        and device.type == "cuda"
        and ffn_kernel_supported(dim, ffn_dim)
    )
