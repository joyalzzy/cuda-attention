# GPU Transformer Optimization — Agent Playbook

This repository is for **Track 3: Implement a GPU Kernel for a Transformer Layer**. The source of truth for competition requirements is `TASK.md`. The executable specifications are:

- `torch_transformer_benchmark.py`
- `tensorflow_transformer_benchmark.py`

When this document disagrees with code, follow the benchmark code and update this document.

## 1. Objective and scope

Build a numerically correct, faster implementation of the supplied Transformer. A submission may target **PyTorch or TensorFlow; one framework is sufficient according to `TASK.md`**. Supporting both is optional and should happen only after one path passes correctness and has a measured GPU speedup.

Primary success criteria:

1. Replace the selected harness's `UserOptimizedTransformer` implementation.
2. Pass the harness's element-wise correctness checks.
3. Measure a speedup over its unmodified baseline on an NVIDIA GPU.
4. Record hardware, software, shapes, dtype, latency, and speedup.
5. Provide the report, repository/README, and demo material requested by `TASK.md`.

Do not claim GPU performance from CPU measurements.

## 2. Strict boundaries

### Allowed

- Change the selected `UserOptimizedTransformer` class.
- Add implementation files under `kernels/`.
- Add profiling and validation utilities under `tools/`.
- Deliberately adapt `copy_model_weights()` if the optimized model has a different parameter layout.
- Add documentation, tests, reports, and reproducibility scripts.

### Not allowed unless the task explicitly requires it

- Do not change `BaselineTransformer`, its blocks, or its attention implementation.
- Do not relax correctness tolerances or alter comparison logic.
- Do not change benchmark timing logic to improve reported numbers.
- Do not skip failing cases and call the remaining subset a complete result.
- Do not report estimated or CPU-only speedups as measured GPU results.

If a benchmark harness itself must be fixed, isolate that change, explain why, and preserve the original comparison semantics.

## 3. Current repository state

| Item | Current state |
|---|---|
| Optimized implementation | PyTorch: Triton kernels (attention/norm/experimental FFN) + SDPA + reference fallback; TensorFlow remains a placeholder |
| `kernels/` | Created: `triton_attention.py`, `triton_norm.py`, `triton_ffn.py` (opt-in), `dispatch.py`; **not yet validated on GPU** |
| `tools/` | PyTorch correctness-matrix, conformance, kernel-gate, and profiler utilities implemented |
| `README.md` | Created with setup, validation, profiling, GPU benchmark instructions, and kernel section |
| Local Python | 3.14.4 virtual environment in `.venv/` |
| Local packages | `.tmp-torch/` holds PyTorch 2.13 + NumPy + Triton used for CPU correctness; `.venv/` itself has only pip |
| Local GPU/CUDA tools | No `nvidia-smi` or `nvcc` found at last verification |
| Local CPU correctness | float32/bfloat16 unmasked, causal, padded, and combined cases pass (kernels disabled on CPU by design) |
| Kernel GPU validation | **pending** — run `tools/test_torch_kernels.py --device cuda` on the target GPU |
| Aspire 2A runner | `run-aspire.sh`: PBS GPU job for kernel gates, CUDA matrix, benchmarks, profiles, and timestamped artifacts; written but never executed locally |

Environment facts can become stale. Re-check them before planning installation or benchmarking.

### Aspire 2A execution policy

- `run-aspire.sh` is a **PBS compute-job script**, not an access script. It contains no `ssh`, `scp`, or `rsync` commands.
- Never execute it on a login node or local workstation; it exits unless `PBS_JOBID` and a readable `PBS_NODEFILE` exist.
- Follow the [workshop Aspire 2A access guide](https://github.com/ntuhpcai/workshops/tree/main/hpc_ai/build_lulesh#accessing-aspire-2a) to reach Aspire 2A, place the checkout in project/scratch storage, then submit with current site values, for example: `qsub -P <PROJECT_CODE> -q <GPU_QUEUE> run-aspire.sh`.
- PBS directives do not reliably expand shell variables. Pass account, queue, resource, and walltime overrides through `qsub` or edit the directives before submission.
- Aspire account codes, queue names, module names, GPU resource syntax, and package-mirror policy are site/account specific; verify them against the current NSCC documentation before submission.
- The script uses a reusable `.venv-aspire`, supports online or wheelhouse installs, captures environment metadata, fails on correctness errors, and writes logs/profiles below `artifacts/aspire2a/`.
- Test order is mandatory: conformance → direct Triton kernel gate → compact CUDA matrix → representative benchmarks → optional profiles. Do not report benchmark output if an earlier gate fails.
- Set `ENABLE_EXPERIMENTAL_FFN=1` only for a separate validation run; the script exports the repository's `TORCH_TRANSFORMER_FFN_KERNEL` variable.

## 4. Exact reference behavior

For input `x` shaped `[B, S, d_model]`, each block is **pre-normalized**:

```text
n1 = LayerNorm(x, eps=1e-5)
attention = MHA(n1)
x = x + attention
n2 = LayerNorm(x, eps=1e-5)
x = x + ffn_out(exact_gelu(ffn_in(n2)))
zero padded query rows, when a valid-token mask is supplied
```

A final `LayerNorm(eps=1e-5)` is applied, followed by another padded-row zeroing operation.

Multi-head attention behavior:

```text
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)
scores = (Q @ K^T) * head_dim**-0.5
scores[future positions] = -inf            # when causal
scores[invalid key positions] = -inf       # when padding mask is supplied
probabilities = softmax(scores in fp32)
context = probabilities cast to input dtype @ V
output = out_proj(context)
output[invalid query rows] = 0              # when padding mask is supplied
```

Required details:

- `d_model % num_heads == 0`.
- Q, K, V, output, FFN input, and FFN output projections use bias.
- GELU is exact: PyTorch `approximate="none"`; TensorFlow `approximate=False`.
- Causal masking removes positions where key index is greater than query index.
- `valid_token_mask` is boolean with shape `[B, S]`.
- All generated test rows have at least one valid key, avoiding all-masked softmax rows.
- Preserve the selected harness's public `forward`/`call` signature and output shape.

## 5. Correctness rules

The two harnesses differ slightly. Do not merge their rules into one approximation.

### PyTorch

For every output element, including non-finite handling implemented by the harness:

```text
abs(user - reference) <= atol
OR
abs(user - reference) <= rtol * abs(reference)
```

Defaults: `atol=0.001`, `rtol=0.01`. Both outputs must be finite at an element for it to pass.

Accuracy failure skips timing unless `--benchmark-on-failure` is supplied. The process returns exit code `2` on accuracy failure.

### TensorFlow

For every output element:

```text
abs(user - reference) < atol
OR
abs(user - reference) <= rtol * abs(reference)
```

Defaults: `atol=0.002`, `rtol=0.02`. Both outputs must be finite at an element for it to pass.

Only passing cases are timed. Failed and error cases cause final exit code `2`; preflight-skipped cases alone do not.

## 6. Harness facts

### PyTorch harness

Default single case:

| B | S | d_model | heads | FFN | layers | dtype |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 128 | 512 | 8 | 2048 | 6 | float32 |

Interface:

```python
class UserOptimizedTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None):
        # Return [B, S, d_model]
```

Important behavior:

- Weights are copied with `load_state_dict(..., strict=True)` by default.
- CUDA timing uses `torch.cuda.Event`; CPU timing uses `perf_counter_ns`.
- Baseline and optimized measurement order alternates by round.
- Reported speedup is baseline median latency divided by optimized median latency.
- TF32 defaults to enabled on CUDA; matmul precision defaults to `high`.

Useful flags:

```text
--device --dtype --causal --padding-ratio
--compile-baseline --compile-user --compile-mode
--matmul-precision --allow-tf32/--no-allow-tf32
--accuracy-trials --rtol --atol
--warmup --repeats --benchmark-rounds
--benchmark-on-failure
```

### TensorFlow harness

Default dimension lists:

```text
batch sizes: [1, 4, 16, 128, 10000]
qkv/d_model: [32, 128, 1024]
heads:       [1, 2, 4, 16]
sequence:    [32, 1024, 100000]
```

These are **not a Cartesian product**. The harness builds compact one-factor-at-a-time cases. Non-sequence sweeps use the shortest sequence. When multiple sequence lengths are configured, the longest one is paired with `B=32`, maximum QKV dimension, and maximum head count as a stress case.

Other defaults: six layers, `ffn_dim = 4 * qkv_dim`, and float16.

Interface:

```python
class UserOptimizedTransformer(BaselineTransformer):
    def call(self, x, valid_token_mask=None, training=False):
        # Return [B, S, qkv_dim]
```

Important behavior:

- In this harness, `qkv_dim` is also `d_model`.
- Strict weight copy checks weight count, order, and shape, then calls `set_weights()`.
- Default execution uses `tf.function`; `--eager` disables it.
- `--compile-baseline` and `--compile-user` enable XLA and cannot be used with `--eager`.
- TensorFlow synchronous execution is enabled so host timing includes completed execution.
- TF32 and GPU memory growth default to enabled.
- A conservative baseline-memory estimate may mark a case `SKIPPED` before model creation.
- Explicit baseline attention is `O(B * H * S^2)`; very long sequences will generally be skipped because the baseline itself is infeasible.
- The default report path is `tensorflow_transformer_benchmark_report.md`.

## 7. Task division for agents

Use one owner per phase. A less-capable model should work on only one phase at a time and must satisfy its exit gate before starting another phase.

### Phase 0 — Select one framework

Owner: project coordinator.

Tasks:

1. Choose PyTorch or TensorFlow as the primary submission path.
2. Record target GPU model and software constraints.
3. Do not begin a second framework until the primary path passes Phases 1–4.

Exit gate: selected framework and target environment are written in `README.md` or a project note.

### Phase 1 — Establish an executable baseline

Owner: environment and validation agent.

Tasks:

1. Verify Python and GPU driver/tool availability.
2. Install versions compatible with the available Python and target GPU. Do not assume Python 3.14 is supported by every framework release. If needed, create a separate supported Python environment instead of forcing incompatible wheels into `.venv/`.
3. Run a tiny correctness smoke test.
4. Run the unmodified representative benchmark on the target GPU.
5. Save commands and raw output.

Exit gate: the selected harness runs end-to-end and baseline latency is recorded.

### Phase 2 — Add the lowest-risk optimized path

Owner: framework implementation agent.

PyTorch first candidate: retain existing parameters but replace explicit attention execution with an equivalent optimized primitive such as scaled-dot-product attention. Consider `torch.compile` separately so its effect can be measured.

TensorFlow first candidate: test graph/XLA improvements before writing a custom op; preserve Keras variables and strict weight-copy behavior.

Tasks:

1. Change only `UserOptimizedTransformer` and new helper files.
2. Support unmasked, causal, padded, and causal-plus-padded cases.
3. Preserve fp32-stable softmax semantics.
4. Add explicit shape/dtype guards and a correct fallback for unsupported cases.

Exit gate: representative correctness passes before performance work continues.

### Phase 3 — Build a correctness matrix

Owner: numerical validation agent.

Minimum matrix for the selected framework:

- dtypes: float32, float16, bfloat16 where the target GPU/framework supports them;
- masks: none, causal, padding, causal plus padding;
- shapes: at least one small, one default/medium, and one larger feasible case;
- more than one random seed.

Tasks:

1. Run accuracy without `--benchmark-on-failure`.
2. Record max absolute error, max relative error, and failed element count.
3. Stop optimization if any supported case fails.

Exit gate: all claimed supported cases pass the harness's exact rule.

### Phase 4 — Profile and optimize

Owner: performance agent.

Optimization order:

1. Measure end-to-end latency and confirm warmup is sufficient.
2. Profile to identify the dominant cost.
3. Optimize only the measured bottleneck.
4. Re-run the Phase 3 correctness matrix after every material change.
5. Use shape dispatch only when measurements justify separate paths.

Potential techniques:

- fused QKV projection;
- optimized/flash-style attention without an `S x S` materialization;
- layout and transpose-copy reduction;
- fused residual plus normalization;
- fused FFN/GELU/residual;
- `torch.compile` or TensorFlow XLA;
- Triton or CUDA only when framework primitives are insufficient.

Exit gate: reproducible GPU speedup on at least one stated shape, with no correctness regression.

### Phase 5 — Reporting and deliverables

Owner: documentation agent.

Tasks:

1. Create `README.md` with setup, exact commands, results, limitations, and team contributions.
2. Record CPU, GPU, memory, driver, CUDA, Python, framework, dtype, shapes, and all relevant flags.
3. Include baseline and optimized median latency and speedup.
4. Explain AI tools used, as requested by `TASK.md`.
5. Prepare the Devpost description, public repository, demo video, and technical report.

Exit gate: a new machine can reproduce the reported result from the documentation.

## 8. Required evidence format

Every performance claim must include:

```text
framework + version:
GPU + driver/CUDA:
Python version:
dtype:
shape: B, S, d_model/qkv_dim, heads, FFN, layers
masking: causal?, padding ratio
compile flags:
TF32/matmul settings:
baseline median ms:
optimized median ms:
speedup:
accuracy: max abs, max relative, failed/total
exact command:
```

Do not use words such as "faster," "optimized," or "supports" without matching evidence.

## 9. Commands

Commands below require compatible packages and, for GPU measurements, a working NVIDIA setup.

```bash
# Inspect environment
.venv/bin/python --version
nvidia-smi
nvcc --version

# PyTorch tiny CPU correctness smoke test
.venv/bin/python torch_transformer_benchmark.py \
  --device cpu --dtype float32 --batch-size 1 --seq-len 8 \
  --d-model 32 --heads 4 --ffn-dim 64 --layers 1 \
  --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1

# PyTorch representative GPU run
.venv/bin/python torch_transformer_benchmark.py \
  --device cuda --dtype float16 --causal --padding-ratio 0.25

# PyTorch compile experiment
.venv/bin/python torch_transformer_benchmark.py \
  --device cuda --dtype bfloat16 --compile-user --compile-mode max-autotune

# TensorFlow small smoke matrix; override defaults to avoid huge cases
.venv/bin/python tensorflow_transformer_benchmark.py \
  --device cpu --dtype float32 --batch-sizes 1 \
  --qkv-dims 32 --heads 4 --seq-lens 8 \
  --ffn-dim 64 --layers 1 --accuracy-trials 1 \
  --warmup 1 --repeats 2 --benchmark-rounds 1

# TensorFlow GPU run with a bounded matrix
.venv/bin/python tensorflow_transformer_benchmark.py \
  --device gpu:0 --dtype float16 --batch-sizes 1 4 16 \
  --qkv-dims 32 128 --heads 1 2 4 --seq-lens 32 1024

# TensorFlow XLA experiment
.venv/bin/python tensorflow_transformer_benchmark.py \
  --device gpu:0 --dtype float16 --compile-user \
  --batch-sizes 1 4 16 --qkv-dims 32 128 \
  --heads 1 2 4 --seq-lens 32 1024
```

## 10. Stop conditions and escalation

Stop and report a blocker instead of guessing when:

- no compatible framework build exists for the active Python version;
- the target GPU or driver is unavailable;
- a requested shape is preflight-skipped because the baseline cannot fit;
- correctness fails and the failing operation is not isolated;
- performance results are CPU-only or too noisy to support a claim.

When blocked, report the exact command, error, environment, and next required action.

## 11. Definition of done

The primary framework is done only when all applicable boxes are checked:

- [ ] Selected `UserOptimizedTransformer` no longer delegates entirely to the baseline.
- [ ] Exact output shape, masking, residual, normalization, GELU, and dtype behavior are preserved.
- [ ] Correctness passes the claimed dtype/mask/shape matrix.
- [ ] Unsupported cases use a correct fallback or fail with a clear message.
- [ ] GPU profiling identifies the bottleneck addressed by the optimization.
- [ ] At least one reproducible GPU speedup is recorded against the unmodified baseline.
- [ ] Results include full environment and command details.
- [ ] `README.md`, technical report, public repository, and demo material meet `TASK.md` requirements.
