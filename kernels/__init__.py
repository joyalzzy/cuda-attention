"""Custom Triton kernels for the optimized Transformer (PyTorch path).

Status: IMPLEMENTED BUT NOT YET VALIDATED ON GPU. These kernels require a
CUDA-capable NVIDIA GPU and a Triton-enabled PyTorch build. They were written
from the reference semantics but have not been compiled or run on this machine
(no NVIDIA GPU / CUDA toolkit available). Validate with
``tools/test_torch_kernels.py`` on the target GPU before trusting them.

Modules:
- ``triton_attention``: flash-style fused multi-head attention (online softmax,
  causal + padding mask support).
- ``triton_norm``: LayerNorm kernel.
- ``triton_ffn``: experimental fused LayerNorm + GEMM1 + GELU + GEMM2 +
  residual + padding-zeroing path (opt-in via env var).
- ``dispatch``: availability checks, shape gating, and kernel selection.
"""

from kernels import dispatch, triton_attention, triton_ffn, triton_norm

__all__ = ["dispatch", "triton_attention", "triton_ffn", "triton_norm"]
