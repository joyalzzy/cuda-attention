#!/usr/bin/env python3
"""Run a compact PyTorch correctness matrix against the benchmark harness.

This tool invokes the public harness instead of duplicating its comparison logic.
It uses minimal timing arguments because its purpose is correctness validation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "torch_transformer_benchmark.py"


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    sequence: int
    model: int
    heads: int
    ffn: int
    causal: bool
    padding: float


CASES = (
    Case("small-unmasked", 1, 16, 32, 4, 64, False, 0.0),
    Case("small-causal", 2, 32, 64, 4, 128, True, 0.0),
    Case("small-padded", 2, 32, 64, 4, 128, False, 0.25),
    Case("medium-causal-padded", 2, 128, 128, 8, 256, True, 0.25),
)


def command_for(case: Case, dtype: str, device: str, trials: int) -> list[str]:
    command = [
        sys.executable,
        str(HARNESS),
        "--device",
        device,
        "--dtype",
        dtype,
        "--batch-size",
        str(case.batch),
        "--seq-len",
        str(case.sequence),
        "--d-model",
        str(case.model),
        "--heads",
        str(case.heads),
        "--ffn-dim",
        str(case.ffn),
        "--layers",
        "2",
        "--padding-ratio",
        str(case.padding),
        "--accuracy-trials",
        str(trials),
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--benchmark-rounds",
        "1",
    ]
    if case.causal:
        command.append("--causal")
    return command


def iter_cases(selected: Iterable[str]) -> Iterable[Case]:
    names = set(selected)
    for case in CASES:
        if not names or case.name in names:
            yield case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtypes", nargs="+", default=["float32"],
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_cases = list(iter_cases(args.cases))
    known = {case.name for case in CASES}
    unknown = set(args.cases) - known
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}; choices={sorted(known)}")
    if args.trials <= 0:
        raise ValueError("--trials must be positive")

    failures = 0
    for dtype in args.dtypes:
        for case in selected_cases:
            command = command_for(case, dtype, args.device, args.trials)
            print(f"\n=== {case.name} | {dtype} ===", flush=True)
            print(" ".join(command), flush=True)
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                failures += 1
                if not args.keep_going:
                    return result.returncode

    print(f"\nvalidation summary: {failures} failing case(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
