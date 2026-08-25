# Technical Report — PyTorch Transformer SDPA Optimization

## Status

CPU correctness has been validated. Final GPU performance numbers must still be recorded on the target machine. No GPU speedup is currently claimed.

## Problem

The supplied baseline implements multi-head attention by explicitly materializing the score matrix, applying fp32 softmax, and multiplying by values. The project must preserve the complete pre-normalized Transformer behavior while reducing GPU latency within the harness's exact element-wise tolerance.

## Selected approach

The primary framework is PyTorch. `UserOptimizedTransformer` preserves all baseline modules and parameters but executes the Transformer through a layered path that prefers **custom Triton kernels**, then SDPA, then the explicit reference:

1. `kernels/triton_attention.py` — flash-style fused multi-head attention:
   online softmax in fp32, causal and padding masks, fp16/bf16 tensor-core
   dots with fp32 accumulation, fp32 dots with `input_precision="ieee"`.
2. `kernels/triton_norm.py` — LayerNorm kernel (fp32 statistics, affine).
3. `kernels/triton_ffn.py` — experimental fused LayerNorm + GEMM1 + exact
   GELU + GEMM2 + residual + padding zeroing (opt-in).
4. `kernels/dispatch.py` — CUDA/Triton availability checks, shape windows,
   and env-var gating (`TORCH_TRANSFORMER_DISABLE_KERNELS`,
   `TORCH_TRANSFORMER_FFN_KERNEL`).
5. SDPA (PyTorch built-in fused attention) and finally the explicit reference
   attention as correctness-first fallbacks. Every kernel entry point is
   guarded by dispatch checks and `try/except` fallbacks.

Preserved semantics include:

- separate biased Q/K/V/output projections;
- `1/sqrt(head_dim)` scaling;
- causal and padding masks;
- zero output for padded query rows;
- pre-normalization and residual ordering;
- exact GELU and final LayerNorm.

**Status of the kernels: implemented but NOT yet validated on GPU.** The
development machine has no NVIDIA GPU or CUDA toolchain, so the Triton kernels
have only been syntax-checked and import-tested. `tools/test_torch_kernels.py`
contains the kernel-vs-reference gate and must pass on the target GPU before
any kernel result is trusted.

## AI-assisted development

AI assistance was used to:

- audit the benchmark scripts as executable specifications;
- distinguish PyTorch and TensorFlow tolerance rules;
- design task phases and exit gates for weaker coding agents;
- implement the SDPA path and validation/profiling utilities;
- write the Triton kernel set (attention, LayerNorm, fused FFN) from the
  reference semantics;
- review uncertainty around mask semantics, backend eligibility, and
  numerical behavior.

All generated changes remain subject to the original harness's correctness
checks, and the Triton kernels remain subject to on-GPU validation.

## Validation plan

`tools/validate_torch.py` invokes the public benchmark harness for:

- unmasked attention;
- causal attention;
- padded attention;
- combined causal and padded attention;
- multiple shapes and random trials;
- float32, float16, and bfloat16 where supported.

The harness reports max absolute error, max relative error, and failing elements. Timing is accepted only after correctness passes.

## Profiling plan

`tools/profile_torch.py` profiles either implementation using identical shapes and dtypes. CPU and CUDA activity, shapes, memory, and optional Chrome traces are captured. The next optimization must target the measured dominant operation rather than an assumed bottleneck.

## Environment and results

### Local CPU validation

Environment: Python 3.14.4; PyTorch 2.13.0+cu130; NumPy 2.5.2; CPU-only (no CUDA device).

Focused conformance suite (`tools/test_torch_conformance.py`): **passed**, covering strict state-dict compatibility, unmasked/causal/padded/combined masking, output dtype and shape, padded-row zeroing, and forced-SDPA-rejection fallback exactness.

Tiny harness runs (`torch_transformer_benchmark.py`, 1–2 layers):

| dtype | masks | accuracy | speedup (CPU, informational) |
|---|---|---|---|
| float32 | none | PASS | 20.7x |
| float32 | causal + padding | PASS | 1.53x |
| bfloat16 | none | PASS | 7.44x |

Compact matrix (`tools/validate_torch.py`): **0 failing cases** across unmasked, causal, padded, and causal-plus-padded float32 shapes; max absolute error at or below 9.54e-7 in all runs.

CPU speedups are informational only. They are not performance evidence and are not claimed as competition results.

### Target GPU results

Fill this section on the target machine:

```text
CPU:
System memory:
GPU:
Driver:
CUDA runtime/build:
Python:
PyTorch:
Dtype:
Shape (B, S, d_model, heads, FFN, layers):
Masking:
TF32/matmul settings:
Exact command:
Accuracy (max abs, max relative, failed/total):
Baseline median latency:
Optimized median latency:
Speedup:
```

## Limitations and next work

- GPU backend selection and performance remain unverified locally.
- The Triton kernels are **not GPU-validated**; run `tools/test_torch_kernels.py --device cuda` first.
- SDPA reduction order may differ from the baseline's explicit fp32 softmax; every claimed configuration must pass the harness.
- Combined causal and padding masking currently materializes an `[S, S]` boolean component and may prevent the best fused backend.
- Kernel tuning (block sizes, `num_warps`, `num_stages`) has not been performed; defaults are used.
- The fused FFN path is experimental and off by default.
- TensorFlow optimization is intentionally deferred because one framework is sufficient under `TASK.md`.
