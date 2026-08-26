#!/usr/bin/env bash
#PBS -N transformer-triton-tests
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l walltime=04:00:00
#PBS -j oe
#
# Aspire 2A PyTorch/Triton validation, benchmark, and profiling job.
#
# This file intentionally contains no remote-access or file-transfer commands.
# Access Aspire 2A separately by following the workshop guide, place this repository in project
# or scratch storage, and submit this file from an Aspire login node with qsub.
#
# Before submission, supply the current NSCC project code and GPU queue through
# qsub (PBS directives do not reliably expand shell variables):
#
#   qsub -P <PROJECT_CODE> -q <GPU_QUEUE> run-aspire.sh
#
# The exact project code, GPU queue, module names, and any revised GPU resource
# syntax are site/account-specific. If needed, override resources at submission:
#
#   qsub -P <PROJECT_CODE> -q <GPU_QUEUE> \
#     -l select=1:ncpus=16:ngpus=1:mem=64gb \
#     -l walltime=04:00:00 run-aspire.sh
#
# Runtime configuration (export before qsub, pass with qsub -v, or edit here):
#   PROJECT_DIR                  checkout path; default PBS_O_WORKDIR
#   VENV_DIR                     reusable venv; default PROJECT_DIR/.venv-aspire
#   PYTHON_MODULE                optional module name; default empty
#   CUDA_MODULE                  optional module name; default empty
#   EXTRA_MODULES                optional space-separated module names
#   INSTALL_DEPS                 auto (default), always, or never
#   PYTORCH_INSTALL_SPEC         pip package spec; default torch
#   PYTORCH_INDEX_URL            optional PyTorch wheel index
#   WHEELHOUSE                   optional directory of offline wheels
#   EXTRA_PIP_ARGS               optional pip flags; never put secrets here
#   CUDA_DTYPES                  space-separated; default all three dtypes
#   ENABLE_EXPERIMENTAL_FFN      0 (default) or 1
#   RUN_PROFILER                 1 (default) or 0
#   PROFILER_DTYPE               default float16
#   HARNESS_WARMUP               default 20
#   HARNESS_REPEATS              default 100
#   HARNESS_ROUNDS               default 3
#   ARTIFACT_ROOT                default PROJECT_DIR/artifacts/aspire2a
#
# The script refuses to run without PBS_JOBID and PBS_NODEFILE, preventing an
# accidental GPU test on a login node or local workstation.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly SCRIPT_NAME="${0##*/}"
readonly START_TIME="$(date -u +%Y%m%dT%H%M%SZ)"
readonly JOB_ID="${PBS_JOBID:-no-pbs-job}"

fatal() {
    printf 'FATAL: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local rc=$?
    printf 'ERROR: %s failed at line %s with exit code %s\n' \
        "$SCRIPT_NAME" "${BASH_LINENO[0]:-unknown}" "$rc" >&2
    exit "$rc"
}
trap on_error ERR

[[ -n "${PBS_JOBID:-}" ]] ||
    fatal "PBS_JOBID is unset; submit this file with qsub on Aspire 2A."
[[ -n "${PBS_NODEFILE:-}" && -r "${PBS_NODEFILE}" ]] ||
    fatal "PBS_NODEFILE is unavailable; no PBS compute allocation was detected."

: "${PROJECT_DIR:=${PBS_O_WORKDIR:-$PWD}}"
: "${VENV_DIR:=${PROJECT_DIR}/.venv-aspire}"
: "${PYTHON_MODULE:=}"
: "${CUDA_MODULE:=}"
: "${EXTRA_MODULES:=}"
: "${INSTALL_DEPS:=auto}"
: "${PYTORCH_INSTALL_SPEC:=torch}"
: "${PYTORCH_INDEX_URL:=}"
: "${WHEELHOUSE:=}"
: "${EXTRA_PIP_ARGS:=}"
: "${CUDA_DTYPES:=float32 float16 bfloat16}"
: "${ENABLE_EXPERIMENTAL_FFN:=0}"
: "${RUN_PROFILER:=1}"
: "${PROFILER_DTYPE:=float16}"
: "${HARNESS_WARMUP:=20}"
: "${HARNESS_REPEATS:=100}"
: "${HARNESS_ROUNDS:=3}"
: "${ARTIFACT_ROOT:=${PROJECT_DIR}/artifacts/aspire2a}"

case "$INSTALL_DEPS" in
    auto|always|never) ;;
    *) fatal "INSTALL_DEPS must be auto, always, or never" ;;
esac
case "$ENABLE_EXPERIMENTAL_FFN" in
    0|1) ;;
    *) fatal "ENABLE_EXPERIMENTAL_FFN must be 0 or 1" ;;
esac
case "$RUN_PROFILER" in
    0|1) ;;
    *) fatal "RUN_PROFILER must be 0 or 1" ;;
esac

[[ -d "$PROJECT_DIR" ]] || fatal "PROJECT_DIR does not exist: $PROJECT_DIR"
for required_file in \
    torch_transformer_benchmark.py \
    tools/test_torch_kernels.py \
    tools/test_torch_conformance.py \
    tools/validate_torch.py \
    tools/profile_torch.py; do
    [[ -f "$PROJECT_DIR/$required_file" ]] ||
        fatal "Missing project file: $PROJECT_DIR/$required_file"
done
cd "$PROJECT_DIR"

# Load only explicitly configured modules. Do not purge: Aspire may inject
# scheduler- or accelerator-related modules into the job environment.
if [[ -n "$PYTHON_MODULE$CUDA_MODULE$EXTRA_MODULES" ]]; then
    type module >/dev/null 2>&1 ||
        fatal "module command is unavailable; clear module variables or initialize modules"
    [[ -z "$PYTHON_MODULE" ]] || module load "$PYTHON_MODULE"
    [[ -z "$CUDA_MODULE" ]] || module load "$CUDA_MODULE"
    if [[ -n "$EXTRA_MODULES" ]]; then
        IFS=' ' read -r -a extra_modules <<< "$EXTRA_MODULES"
        module load "${extra_modules[@]}"
    fi
fi

command -v python3 >/dev/null 2>&1 ||
    fatal "python3 unavailable; set PYTHON_MODULE to an Aspire-provided module"
command -v nvidia-smi >/dev/null 2>&1 ||
    fatal "nvidia-smi unavailable; this does not appear to be an NVIDIA GPU node"
nvidia-smi -L | grep -q '^GPU ' || fatal "No allocated NVIDIA GPU is visible"

# Keep the reusable environment in project/scratch storage. Do not remove or
# overwrite a non-venv directory.
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    if [[ -e "$VENV_DIR" ]]; then
        [[ -d "$VENV_DIR" ]] || fatal "VENV_DIR exists but is not a directory: $VENV_DIR"
        [[ -z "$(find "$VENV_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] ||
            fatal "VENV_DIR is non-empty but is not a usable venv: $VENV_DIR"
    else
        mkdir -p "$VENV_DIR"
    fi
    python3 -m venv "$VENV_DIR"
fi

readonly PYTHON="$VENV_DIR/bin/python"
[[ -x "$PYTHON" ]] || fatal "Invalid venv: $VENV_DIR"

# Persistent caches reduce repeated Triton compilation and downloads. A
# job-specific Triton cache avoids concurrent-writer corruption.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_DIR}/.cache/aspire2a}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${XDG_CACHE_HOME}/triton/${JOB_ID}}"
mkdir -p "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

need_install=0
if ! "$PYTHON" - <<'PY'
import numpy
import torch
import triton
assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
print("torch", torch.__version__)
print("triton", triton.__version__)
print("numpy", numpy.__version__)
PY
then
    need_install=1
fi

if [[ "$INSTALL_DEPS" == always ||
      ( "$INSTALL_DEPS" == auto && "$need_install" == 1 ) ]]; then
    "$PYTHON" -m pip install --upgrade pip setuptools wheel

    pip_args=(install --upgrade "$PYTORCH_INSTALL_SPEC" numpy)
    if [[ -n "$WHEELHOUSE" ]]; then
        [[ -d "$WHEELHOUSE" ]] || fatal "WHEELHOUSE does not exist: $WHEELHOUSE"
        pip_args+=(--no-index --find-links "$WHEELHOUSE")
    elif [[ -n "$PYTORCH_INDEX_URL" ]]; then
        pip_args+=(--index-url "$PYTORCH_INDEX_URL")
    fi
    if [[ -n "$EXTRA_PIP_ARGS" ]]; then
        IFS=' ' read -r -a extra_pip_args <<< "$EXTRA_PIP_ARGS"
        pip_args+=("${extra_pip_args[@]}")
    fi
    "$PYTHON" -m pip "${pip_args[@]}"

    # Most CUDA PyTorch wheels include a compatible Triton. Install it
    # separately only if the chosen wheel did not provide it.
    if ! "$PYTHON" -c 'import triton' >/dev/null 2>&1; then
        triton_args=(install --upgrade triton)
        if [[ -n "$WHEELHOUSE" ]]; then
            triton_args+=(--no-index --find-links "$WHEELHOUSE")
        fi
        "$PYTHON" -m pip "${triton_args[@]}"
    fi
elif [[ "$need_install" == 1 ]]; then
    fatal "PyTorch/Triton/NumPy/CUDA check failed and INSTALL_DEPS=never"
fi

# Final dependency and GPU gate before collecting any result.
"$PYTHON" - <<'PY'
import numpy, torch, triton
assert torch.cuda.is_available(), "CUDA unavailable to PyTorch"
assert torch.cuda.device_count() >= 1, "No CUDA devices visible"
print("Python package check")
print("  torch:", torch.__version__)
print("  torch CUDA runtime:", torch.version.cuda)
print("  triton:", triton.__version__)
print("  numpy:", numpy.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}; capability={p.major}.{p.minor}; "
          f"memory={p.total_memory / 2**30:.2f} GiB")
PY

safe_job_id="${JOB_ID//[^[:alnum:]._-]/_}"
readonly RUN_DIR="${ARTIFACT_ROOT}/${START_TIME}_${safe_job_id}"
mkdir -p "$RUN_DIR"/{environment,logs,profiles}
ln -sfn "$RUN_DIR" "$ARTIFACT_ROOT/latest" 2>/dev/null || true

export TORCH_TRANSFORMER_DISABLE_KERNELS=0
export TORCH_TRANSFORMER_FFN_KERNEL="$ENABLE_EXPERIMENTAL_FFN"

# Capture reproducibility metadata. Optional probes must not abort the job.
{
    printf 'UTC start: %s\n' "$START_TIME"
    printf 'PBS_JOBID: %s\n' "$JOB_ID"
    printf 'PBS_QUEUE: %s\n' "${PBS_QUEUE:-unknown}"
    printf 'PBS_O_WORKDIR: %s\n' "${PBS_O_WORKDIR:-unknown}"
    printf 'Host: %s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'Project directory: %s\n' "$PROJECT_DIR"
    printf 'Venv: %s\n' "$VENV_DIR"
    printf 'Experimental FFN: %s\n' "$ENABLE_EXPERIMENTAL_FFN"
    printf '\nAllocated nodes:\n'
    sort -u "$PBS_NODEFILE"
    printf '\nEnvironment (credential-like names removed):\n'
    env | LC_ALL=C sort |
        sed -E '/(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|COOKIE|AUTH|AWS_|AZURE_|GITHUB_)/Id'
} > "$RUN_DIR/environment/job.txt"

(type module >/dev/null 2>&1 && module -t list 2>&1 || true) \
    > "$RUN_DIR/environment/modules.txt"
(nvidia-smi || true) > "$RUN_DIR/environment/nvidia-smi.txt" 2>&1
(nvidia-smi -q || true) > "$RUN_DIR/environment/nvidia-smi-query.txt" 2>&1
(nvcc --version || true) > "$RUN_DIR/environment/nvcc.txt" 2>&1
"$PYTHON" --version > "$RUN_DIR/environment/python.txt" 2>&1
"$PYTHON" -m pip freeze > "$RUN_DIR/environment/pip-freeze.txt"

"$PYTHON" - <<'PY' > "$RUN_DIR/environment/torch.txt"
import platform, torch, triton
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

run_logged() {
    local name=$1
    shift
    printf '\n[%s] START %s\n' "$(date -u +%FT%TZ)" "$name" |
        tee -a "$RUN_DIR/logs/summary.log"
    printf 'Command:' >> "$RUN_DIR/logs/summary.log"
    printf ' %q' "$@" >> "$RUN_DIR/logs/summary.log"
    printf '\n' >> "$RUN_DIR/logs/summary.log"
    "$@" 2>&1 | tee "$RUN_DIR/logs/${name}.log"
    printf '[%s] PASS %s\n' "$(date -u +%FT%TZ)" "$name" |
        tee -a "$RUN_DIR/logs/summary.log"
}

IFS=' ' read -r -a dtype_list <<< "$CUDA_DTYPES"
((${#dtype_list[@]} > 0)) || fatal "CUDA_DTYPES must contain at least one dtype"

# Gate 1: CPU/reference conformance. This tool currently has no device/dtype
# arguments and deliberately exercises strict copy + forced fallback behavior.
run_logged conformance "$PYTHON" tools/test_torch_conformance.py

# Gate 2: compile and validate every custom Triton kernel directly on the GPU.
# This must pass before any benchmark result is accepted.
run_logged kernels "$PYTHON" tools/test_torch_kernels.py \
    --device cuda --dtypes "${dtype_list[@]}"

# Gate 3: compact end-to-end CUDA matrix. The tool invokes the public harness
# and stops on the first failure because --keep-going is intentionally omitted.
run_logged validation "$PYTHON" tools/validate_torch.py \
    --device cuda --dtypes "${dtype_list[@]}" --trials 3

common_benchmark_args=(
    --device cuda
    --batch-size 8
    --seq-len 128
    --d-model 512
    --heads 8
    --ffn-dim 2048
    --layers 6
    --accuracy-trials 5
    --warmup "$HARNESS_WARMUP"
    --repeats "$HARNESS_REPEATS"
    --benchmark-rounds "$HARNESS_ROUNDS"
)

# Representative end-to-end matrix: unmasked, causal, padding, and combined.
# Never pass --benchmark-on-failure; the harness exits 2 on correctness failure.
for dtype in "${dtype_list[@]}"; do
    run_logged "benchmark_${dtype}_unmasked" \
        "$PYTHON" torch_transformer_benchmark.py \
        "${common_benchmark_args[@]}" --dtype "$dtype"

    run_logged "benchmark_${dtype}_causal" \
        "$PYTHON" torch_transformer_benchmark.py \
        "${common_benchmark_args[@]}" --dtype "$dtype" --causal

    run_logged "benchmark_${dtype}_padding" \
        "$PYTHON" torch_transformer_benchmark.py \
        "${common_benchmark_args[@]}" --dtype "$dtype" --padding-ratio 0.25

    run_logged "benchmark_${dtype}_combined" \
        "$PYTHON" torch_transformer_benchmark.py \
        "${common_benchmark_args[@]}" --dtype "$dtype" \
        --causal --padding-ratio 0.25
done

if [[ "$RUN_PROFILER" == 1 ]]; then
    profile_args=(
        --device cuda
        --dtype "$PROFILER_DTYPE"
        --batch-size 8
        --seq-len 128
        --d-model 512
        --heads 8
        --ffn-dim 2048
        --layers 6
        --warmup 5
        --steps 10
    )

    run_logged profile_baseline \
        "$PYTHON" tools/profile_torch.py \
        --model baseline "${profile_args[@]}" \
        --trace "$RUN_DIR/profiles/baseline.json"

    run_logged profile_optimized \
        "$PYTHON" tools/profile_torch.py \
        --model optimized "${profile_args[@]}" \
        --trace "$RUN_DIR/profiles/optimized.json"

    run_logged profile_optimized_combined \
        "$PYTHON" tools/profile_torch.py \
        --model optimized "${profile_args[@]}" \
        --causal --padding-ratio 0.25 \
        --trace "$RUN_DIR/profiles/optimized-combined.json"
fi

printf '\nAll requested Aspire 2A checks passed at %s.\nArtifacts: %s\n' \
    "$(date -u +%FT%TZ)" "$RUN_DIR" |
    tee -a "$RUN_DIR/logs/summary.log"
