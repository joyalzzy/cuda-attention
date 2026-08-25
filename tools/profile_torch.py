#!/usr/bin/env python3
"""Profile the baseline or optimized PyTorch Transformer with torch.profiler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
    generate_random_case,
    resolve_device,
    resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--trace", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.steps <= 0:
        raise ValueError("--warmup must be non-negative and --steps positive")
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("--padding-ratio must be in [0, 1)")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )
    config.validate()

    torch.manual_seed(1234)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(baseline, optimized)
    model = baseline if args.model == "baseline" else optimized
    model = model.to(device=device, dtype=dtype).eval()
    x, mask = generate_random_case(config, device, dtype, 1234, args.padding_ratio, 1.0)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode(), torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        for _ in range(args.steps):
            model(x, mask)
            # Synchronize inside the profiled region so all device work is
            # captured before the profiler context closes.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            profile.step()

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(profile.key_averages().table(sort_by=sort_key, row_limit=args.row_limit))
    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))
        print(f"Chrome trace written to {args.trace.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
