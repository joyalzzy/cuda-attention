#!/usr/bin/env python3
"""Focused conformance tests for the PyTorch optimized Transformer."""

from __future__ import annotations

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
    compare_outputs,
    copy_model_weights,
)


def make_config(causal: bool) -> TransformerConfig:
    return TransformerConfig(2, 12, 32, 4, 64, 2, causal)


def test_case(causal: bool, padded: bool) -> None:
    config = make_config(causal)
    baseline = BaselineTransformer(config).eval()
    optimized = UserOptimizedTransformer(config).eval()
    assert baseline.state_dict().keys() == optimized.state_dict().keys()
    copy_model_weights(baseline, optimized, strict=True)

    generator = torch.Generator().manual_seed(2025 + int(causal) + 2 * int(padded))
    x = torch.randn(config.batch_size, config.seq_len, config.d_model, generator=generator)
    mask = torch.ones(config.batch_size, config.seq_len, dtype=torch.bool)
    if padded:
        mask[0, -3:] = False
        mask[1, -1:] = False
        x = x.masked_fill(~mask[..., None], 0)

    with torch.inference_mode():
        reference = baseline(x, mask)
        candidate = optimized(x, mask)
    result = compare_outputs(reference, candidate, rtol=0.01, atol=0.001)
    assert result.passed, result
    assert candidate.dtype == x.dtype
    assert candidate.shape == x.shape
    if padded:
        expanded_invalid_rows = (~mask)[..., None].expand_as(candidate)
        assert torch.count_nonzero(candidate.masked_select(expanded_invalid_rows)) == 0


def test_forced_fallback() -> None:
    config = make_config(causal=True)
    baseline = BaselineTransformer(config).eval()
    optimized = UserOptimizedTransformer(config).eval()
    copy_model_weights(baseline, optimized)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)
    mask = torch.ones(config.batch_size, config.seq_len, dtype=torch.bool)
    mask[:, -2:] = False
    x = x.masked_fill(~mask[..., None], 0)

    original = torch.nn.functional.scaled_dot_product_attention
    try:
        def reject_sdpa(*args: object, **kwargs: object) -> torch.Tensor:
            raise RuntimeError("forced backend rejection")

        torch.nn.functional.scaled_dot_product_attention = reject_sdpa
        with torch.inference_mode():
            reference = baseline(x, mask)
            candidate = optimized(x, mask)
    finally:
        torch.nn.functional.scaled_dot_product_attention = original

    assert torch.equal(reference, candidate), "fallback must match baseline exactly"


def main() -> int:
    for causal in (False, True):
        for padded in (False, True):
            test_case(causal, padded)
    test_forced_fallback()
    print("PyTorch conformance tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
