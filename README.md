# GPU Transformer Optimization

Primary implementation path: **PyTorch**. The optimized model uses PyTorch scaled-dot-product attention (SDPA), allowing CUDA builds to select fused Flash Attention or memory-efficient kernels when supported. The original module hierarchy and parameters are preserved for strict weight copying. TensorFlow remains an unmodified optional harness.

## Current status

- PyTorch `UserOptimizedTransformer`: implemented with SDPA **plus custom Triton kernels**.
- Custom Triton kernels in `kernels/`:
  - `triton_attention.py` — flash-style fused multi-head attention (online softmax, causal + padding mask).
  - `triton_norm.py` — LayerNorm kernel.
  - `triton_ffn.py` — experimental fused LayerNorm+GEMM1+GELU+GEMM2+residual (opt-in).
  - `dispatch.py` — availability checks, shape gating, env-var controls.
- **Kernels are NOT yet validated on GPU** (no NVIDIA GPU / CUDA toolchain on this machine). They are written from the reference semantics, syntax-checked, and import-tested, but have not been compiled or run. Run `tools/test_torch_kernels.py` on the target GPU before trusting them.
- Correctness fallback chain: Triton kernel → SDPA → explicit reference attention. Any kernel failure or unsupported shape falls back automatically.
- Conformance suite: `tools/test_torch_conformance.py` passes.
- CPU correctness: validated on float32 and bfloat16 across unmasked, causal, padded, and combined-mask cases (kernels disabled on CPU by design).

CPU latency is not performance evidence. No speedup is claimed until the GPU procedure below has been run and its output recorded.

## Reference behavior preserved

The implementation retains:

- pre-normalized Transformer blocks;
- separate biased Q/K/V/output and FFN projections;
- exact GELU;
- causal masking;
- invalid-key padding masks and padded-query output zeroing;
- final LayerNorm;
- strict-compatible state-dict names and shapes.

SDPA may use a different reduction order than the explicit fp32-softmax baseline. The supplied harness remains the authority for whether numerical differences pass tolerance.

## Environment setup

Use a Python version supported by the desired PyTorch release. The repository's original `.venv` uses Python 3.14. A compatible PyTorch 2.13 wheel was discoverable during local validation, but target-machine support and the desired CUDA wheel must still be checked against current PyTorch installation documentation. Python 3.11 or 3.12 remains a conservative fallback for older releases.

```bash
python3.12 -m venv .venv-torch
source .venv-torch/bin/activate
python -m pip install --upgrade pip

# Choose the command matching the target CUDA runtime at:
# https://pytorch.org/get-started/locally/
pip install torch
```

Verify the accelerator before benchmarking:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
nvidia-smi
```

## Correctness validation

Tiny direct harness run:

```bash
python torch_transformer_benchmark.py \
  --device cpu --dtype float32 --batch-size 1 --seq-len 16 \
  --d-model 32 --heads 4 --ffn-dim 64 --layers 1 \
  --accuracy-trials 2 --warmup 1 --repeats 2 --benchmark-rounds 1
```

Focused state-dict, mask, SDPA, and forced-fallback conformance tests:

```bash
python tools/test_torch_conformance.py
```

Kernel validation (the GPU gate for `kernels/`):

```bash
python tools/test_torch_kernels.py --device cuda --dtypes float32 float16 bfloat16
```

Compact matrix on CPU:

```bash
python tools/validate_torch.py --device cpu --dtypes float32
```

GPU matrix, where supported:

```bash
python tools/validate_torch.py \
  --device cuda --dtypes float32 float16 bfloat16 --keep-going
```

The matrix covers unmasked, causal, padded, and causal-plus-padded cases. Add benchmark-specific shapes before claiming broad support.

## Benchmarking

Representative run:

```bash
python torch_transformer_benchmark.py \
  --device cuda --dtype float16 \
  --batch-size 8 --seq-len 128 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 \
  --accuracy-trials 5 --warmup 20 --repeats 100 --benchmark-rounds 3
```

Mask variants:

```bash
# Causal
python torch_transformer_benchmark.py --device cuda --dtype float16 --causal

# Padding
python torch_transformer_benchmark.py \
  --device cuda --dtype float16 --padding-ratio 0.25

# Causal + padding
python torch_transformer_benchmark.py \
  --device cuda --dtype float16 --causal --padding-ratio 0.25
```

Compilation is a separate experiment and should not be mixed into the first SDPA comparison:

```bash
python torch_transformer_benchmark.py \
  --device cuda --dtype float16 \
  --compile-user --compile-mode max-autotune
```

## Profiling

Profile each implementation with identical arguments:

```bash
python tools/profile_torch.py \
  --model baseline --device cuda --dtype float16 \
  --trace artifacts/baseline-trace.json

python tools/profile_torch.py \
  --model optimized --device cuda --dtype float16 \
  --trace artifacts/optimized-trace.json
```

Use the tables and traces to decide whether attention, projections, FFN, normalization, memory layout, or launch overhead is the next bottleneck. Do not add custom CUDA/Triton kernels before profiling justifies them.

## Results template

Record every performance claim in this form:

```text
framework + version:
GPU + driver/CUDA:
CPU and system memory:
Python version:
dtype:
shape: B, S, d_model, heads, FFN, layers
masking: causal?, padding ratio
compile flags:
TF32/matmul settings:
baseline median ms:
optimized median ms:
speedup:
accuracy: max abs, max relative, failed/total
exact command:
```

## Custom Triton kernels

The competition deliverable is a custom GPU kernel. `kernels/` provides a Triton-based kernel set (Triton ships with PyTorch ≥ 2.1 CUDA builds, so no separate CUDA toolkit is needed to compile them):

| File | Kernel | Status |
|---|---|---|
| `kernels/triton_attention.py` | flash-style fused multi-head attention: online softmax in fp32, causal + padding masks, fp16/bf16 tensor-core dots, fp32 `ieee` dots | implemented, **not GPU-validated** |
| `kernels/triton_norm.py` | LayerNorm (fp32 statistics, affine) | implemented, **not GPU-validated** |
| `kernels/triton_ffn.py` | experimental fused LayerNorm + GEMM1 + exact GELU + GEMM2 + residual + padding zeroing (two kernels) | implemented, **not GPU-validated**, opt-in |
| `kernels/dispatch.py` | availability checks, shape windows, env-var gating | validated (logic) |

Dispatch policy:

- Kernels run **only on CUDA** with supported shapes.
- Attention: `kernels/` → SDPA → reference, with `try/except` fallback at every step.
- LayerNorm: kernel on CUDA within the shape window.
- Fused FFN: **opt-in** via `TORCH_TRANSFORMER_FFN_KERNEL=1` (least mature; the GEMM tiling trades compute for fewer launches).
- `TORCH_TRANSFORMER_DISABLE_KERNELS=1` disables all kernels (forces SDPA/reference).
- Missing or broken Triton imports degrade silently to the framework path (verified).

GPU validation procedure:

```bash
# 1. Kernel-vs-reference gate (must pass before benchmarking)
python tools/test_torch_kernels.py --device cuda --dtypes float32 float16 bfloat16

# 2. Correctness matrix with kernels enabled (default on CUDA)
python tools/validate_torch.py --device cuda --dtypes float32 float16 bfloat16 --keep-going

# 3. Benchmark
python torch_transformer_benchmark.py --device cuda --dtype float16 --causal --padding-ratio 0.25

# 4. Profile to see whether the kernels are actually the bottleneck
python tools/profile_torch.py --model optimized --device cuda --dtype float16 --trace artifacts/optimized-trace.json
```

## Known limitations

- GPU correctness, kernel selection, and speedup are unverified in this checkout; the Triton kernels must pass `tools/test_torch_kernels.py` on the target GPU before any claim.
- The flash-attention kernel requires head dimension ≤ 256 and sequence length ≥ 16 (dispatch falls back otherwise).
- The fused FFN path is experimental and disabled by default; enable it only after profiling shows the FFN is a measured bottleneck.
- Combined causal and padding masking uses an explicit additive `[S, S]` causal component inside the kernel. This is correct but can limit fused backend selection and is unsuitable for extreme sequence lengths. All-True harness masks are detected and use the mask-free/`is_causal` path. Profiling on the target GPU should determine whether a custom compact-mask path is necessary.
- SDPA backend selection depends on PyTorch version, GPU architecture, dtype, head dimension, masks, and runtime settings.
- The project's Triton kernels are not yet tuned (block sizes, num_warps, num_stages are defaults); tuning is a follow-up after GPU profiling.
- The TensorFlow optimized placeholder is unchanged because the competition requires only one framework path.

## Deliverables still requiring project-owner input

- Target GPU and final environment specification.
- Measured correctness and timing output.
- Devpost narrative and public repository URL.
- Demo video URL.
- Team member contributions and final technical report.
