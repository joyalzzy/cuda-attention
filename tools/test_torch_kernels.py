#!/usr/bin/env python3
"""Validate the custom Triton kernels against the reference implementation.

On machines without CUDA the kernel-vs-reference tests are skipped and only
the fallback-path guarantees are checked. On a CUDA machine this script is the
primary gate before trusting any kernel: every kernel must match the reference
within harness tolerances.

Usage:
    python tools/test_torch_kernels.py                # CPU: fallback checks
    python tools/test_torch_kernels.py --dtypes float32 float16 bfloat16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import torch_transformer_benchmark as tb
from kernels import dispatch
from kernels import triton_attention, triton_ffn, triton_norm


def make_data(batch, heads, seq, dim, dtype, device, seed, causal, padded):
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(batch, heads, seq, dim, generator=generator, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, dim, generator=generator, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, dim, generator=generator, device=device, dtype=dtype)
    mask = torch.ones(batch, seq, device=device, dtype=torch.bool)
    if padded:
        mask[0, seq // 2:] = False
        mask[-1, -1:] = False
    return q, k, v, mask


def reference_attention(q, k, v, mask, causal, scale):
    """Explicit mirror of the harness attention math (fp32 softmax)."""
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        seq = q.shape[2]
        causal_mask = torch.ones(seq, seq, device=q.device, dtype=torch.bool).triu(
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probs, v)


def test_fallback_paths():
    """CPU: kernels must never run, and the model must match the baseline."""
    assert dispatch.triton_available() is False or not torch.cuda.is_available()
    config = tb.TransformerConfig(2, 16, 32, 4, 64, 2, True)
    baseline = tb.BaselineTransformer(config).eval()
    optimized = tb.UserOptimizedTransformer(config).eval()
    tb.copy_model_weights(baseline, optimized)
    x = torch.randn(2, 16, 32)
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, -2:] = False
    with torch.inference_mode():
        ref = baseline(x, mask)
        cand = optimized(x, mask)
    result = tb.compare_outputs(ref, cand, rtol=0.01, atol=0.001)
    assert result.passed, f"CPU optimized path failed tolerances: {result}"
    print("fallback-path test passed")


def test_kernels(device, dtypes):
    torch.manual_seed(0)
    config = tb.TransformerConfig(2, 64, 64, 4, 128, 2, False)
    baseline = tb.BaselineTransformer(config).eval().to(device)
    optimized = tb.UserOptimizedTransformer(config).eval().to(device)
    tb.copy_model_weights(baseline, optimized)

    for dtype in dtypes:
        for causal in (False, True):
            for padded in (False, True):
                q, k, v, mask = make_data(2, 4, 64, 16, dtype, device, 7, causal, padded)
                scale = 16.0**-0.5
                kernel_out = triton_attention.flash_attention(
                    q, k, v, mask if padded else None, causal, scale
                )
                ref_out = reference_attention(q, k, v, mask, causal, scale)
                diff = (kernel_out.float() - ref_out.float()).abs()
                rel = diff / ref_out.float().abs().clamp_min(1e-12)
                passed = (diff <= 0.001) | (rel <= 0.01)
                if not bool(passed.all()):
                    raise AssertionError(
                        f"attention kernel mismatch dtype={dtype} causal={causal} "
                        f"padded={padded} max_abs={diff.max().item():.6g}"
                    )
                if padded:
                    invalid = ~mask[..., None]
                    assert bool((kernel_out.masked_select(invalid.expand_as(kernel_out)) == 0).all())

        # LayerNorm kernel vs torch LayerNorm.
        x = torch.randn(2, 64, 64, device=device, dtype=dtype)
        weight = torch.randn(64, device=device, dtype=dtype)
        bias = torch.randn(64, device=device, dtype=dtype)
        ln_ref = torch.nn.functional.layer_norm(x, (64,), weight, bias, 1e-5)
        ln_kernel = triton_norm.layernorm(x, weight, bias, 1e-5)
        diff = (ln_kernel.float() - ln_ref.float()).abs()
        rel = diff / ln_ref.float().abs().clamp_min(1e-12)
        assert bool(((diff <= 0.001) | (rel <= 0.01)).all()), f"layernorm mismatch {dtype}"

        # Fused FFN vs torch reference (with and without padding).
        for padded in (False, True):
            x = torch.randn(2, 64, 64, device=device, dtype=dtype)
            norm = torch.nn.LayerNorm(64, eps=1e-5).to(device=device, dtype=dtype)
            ffn_in = torch.nn.Linear(64, 128, bias=True).to(device=device, dtype=dtype)
            ffn_out = torch.nn.Linear(128, 64, bias=True).to(device=device, dtype=dtype)
            mask = torch.ones(2, 64, device=device, dtype=torch.bool)
            if padded:
                mask[:, -8:] = False
            ref = ffn_out(torch.nn.functional.gelu(ffn_in(norm(x)), approximate="none")) + x
            if padded:
                ref = ref.masked_fill(~mask[..., None], 0)
            kernel = triton_ffn.fused_ffn(x, norm, ffn_in, ffn_out, mask if padded else None)
            diff = (kernel.float() - ref.float()).abs()
            rel = diff / ref.float().abs().clamp_min(1e-12)
            assert bool(((diff <= 0.001) | (rel <= 0.01)).all()), (
                f"ffn mismatch dtype={dtype} padded={padded} max_abs={diff.max().item():.6g}"
            )

        # End-to-end model with kernels enabled via env vars is covered by
        # tools/validate_torch.py on CUDA; here we verify kernel dispatch flags.
        assert dispatch.choose_attention(device, 64, 16, dtype) == "triton"
        assert dispatch.choose_layernorm(device, 64) is True

    print(f"kernel-vs-reference tests passed on {device} for {[d.name for d in dtypes]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtypes", nargs="+", default=["float32", "float16", "bfloat16"],
        choices=("float32", "float16", "bfloat16"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_fallback_paths()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
        test_kernels(device, [getattr(torch, name) for name in args.dtypes])
    elif args.device.startswith("cuda"):
        raise SystemExit("CUDA requested but unavailable; cannot run kernel tests")
    else:
        print("no CUDA device: kernel-vs-reference tests skipped (run on the target GPU)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
