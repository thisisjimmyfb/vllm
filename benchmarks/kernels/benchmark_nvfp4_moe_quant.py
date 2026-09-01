# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the NVFP4 MoE-experts quantization kernels: vLLM vs FlashInfer.

Unlike benchmark_nvfp4_quant.py (dense `scaled_fp4_quant`) or
benchmark_cutlass_moe_nvfp4.py (whole cutlass_moe_fp4 block, including
GEMMs), this isolates just the per-expert quantization kernel for each
backend:

  vLLM (ragged/packed layout, CUTLASS grouped-GEMM convention):
    - scaled_fp4_experts_quant
    - silu_and_mul_scaled_fp4_experts_quant

  FlashInfer (batched/masked layout, one padded [B, M_max, K] tensor):
    - scaled_fp4_grouped_quantize
    - silu_and_mul_scaled_nvfp4_experts_quantize

Both backends are fed the *same* per-expert token counts (derived once from
a random routing via get_cutlass_moe_mm_data) so the comparison reflects
real backend/kernel differences rather than differing workloads. Note the
two backends fundamentally differ in layout: vLLM operates on exactly
sum(counts) ragged rows, while FlashInfer pads every expert up to
max(counts) rows (plus internal 128-row tiling) -- so some of FlashInfer's
extra time on skewed routing is inherent padding overhead, not raw kernel
speed. That skew sensitivity is itself part of what this benchmark is
meant to surface.
"""

import argparse
import os

import torch

# vLLM's scaled_fp4_experts_quant/silu_and_mul_scaled_fp4_experts_quant size
# their output_scales buffer to VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk
# rows -- a fixed worst-case bound (default 163840) chosen so the op never
# needs a device->host sync to learn the real per-expert row count. That's
# ~5x larger than this benchmark's max swept num_tokens (32768), so left at
# its default the vLLM side pays a huge, unrepresentative allocation on every
# call while FlashInfer's side (sized to the real per-expert max, computed
# via an explicit sync in _route() below) does not -- making the two
# providers' latencies incomparable. Bound it to this benchmark's actual max
# num_tokens instead, which is still a safe upper bound (a single expert can
# receive at most num_tokens rows: each of the num_tokens original rows
# contributes to at most one copy per distinct expert it's routed to). Keep
# this in sync with the max of x_vals below if that sweep range changes.
os.environ.setdefault("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", "32768")

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.flashinfer import (
    scaled_fp4_grouped_quantize as flashinfer_scaled_fp4_grouped_quantize,
)
from vllm.utils.flashinfer import (
    silu_and_mul_scaled_nvfp4_experts_quantize as flashinfer_silu_and_mul_quantize,
)

if not current_platform.has_device_capability(100):
    raise RuntimeError("NVFP4 requires compute capability of 10.0 (Blackwell)")

PROVIDERS = ["vllm", "flashinfer"]
GLOBAL_SF = 448.0 * 6.0  # FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NVFP4 MoE-experts quantization: vLLM vs FlashInfer"
    )
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument(
        "--providers", nargs="+", type=str, default=PROVIDERS, choices=PROVIDERS
    )
    parser.add_argument("--save-path", type=str, default=None)
    return parser.parse_args()


args = _parse_args()


def _route(num_tokens: int, e: int, n: int, k: int, topk: int, device: str):
    """Build a random routing once, return vLLM offsets + per-expert counts."""
    gating = torch.randn(num_tokens, e, device=device)
    topk_ids = torch.topk(gating, topk, dim=-1).indices.to(torch.int32)

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

    ops.get_cutlass_moe_mm_data(
        topk_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        e,
        n,
        k,
        blockscale_offsets,
        is_gated=True,
    )
    counts = (expert_offsets[1:] - expert_offsets[:-1]).clamp(min=1)
    return a_map, expert_offsets, blockscale_offsets, counts


def _make_benchmark(op_name: str, gated: bool):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens"],
            # Coarse on purpose: skip the fine low-end granularity, but keep
            # 256 explicit since that's DiffusionGemma's canvas_length (a
            # full-canvas diffusion decode step, not a per-token AR step --
            # see the kLoadsPerIteration profiling discussion). The upper end
            # goes past the old 8192 ceiling to cover larger, real-model-scale
            # batches too.
            x_vals=[32, 256, 1024, 2048, 4096, 8192, 16384, 32768],
            x_log=False,
            line_arg="provider",
            line_vals=args.providers,
            line_names=args.providers,
            ylabel="us (lower is better)",
            plot_name=f"NVFP4 MoE-Experts {op_name} Latency (us)",
            args={},
        )
    )
    def benchmark(num_tokens, provider, e, topk, hidden_size, intermediate_size):
        device = "cuda"
        dtype = torch.bfloat16
        k = hidden_size
        n = intermediate_size
        width = 2 * n if gated else k

        a_map, expert_offsets, blockscale_offsets, counts = _route(
            num_tokens, e, n, k, topk, device
        )
        quantiles = [0.5, 0.2, 0.8]

        if provider == "vllm":
            m_topk = a_map.numel()
            gscale = torch.full((e,), GLOBAL_SF, dtype=torch.float32, device=device)
            if gated:
                a = torch.randn(m_topk, width, dtype=dtype, device=device)
                fn = lambda: ops.silu_and_mul_scaled_fp4_experts_quant(
                    a, gscale, expert_offsets, blockscale_offsets, topk
                )
            else:
                hidden_states = torch.randn(num_tokens, k, dtype=dtype, device=device)
                a = ops.shuffle_rows(hidden_states, a_map)
                fn = lambda: ops.scaled_fp4_experts_quant(
                    a, gscale, expert_offsets, blockscale_offsets, topk
                )
        else:
            m_max = int(counts.max().item())
            a_batched = torch.randn(e, m_max, width, dtype=dtype, device=device)
            mask = counts.to(torch.int32)
            gscale = torch.full((e,), GLOBAL_SF, dtype=torch.float32, device=device)
            if gated:
                fn = lambda: flashinfer_silu_and_mul_quantize(a_batched, mask, gscale)
            else:
                fn = lambda: flashinfer_scaled_fp4_grouped_quantize(
                    a_batched, mask, gscale
                )

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
        return ms * 1e3, max_ms * 1e3, min_ms * 1e3

    return benchmark


benchmark_quant = _make_benchmark("scaled_fp4_experts_quant", gated=False)
benchmark_silu_mul = _make_benchmark(
    "silu_and_mul_scaled_fp4_experts_quant", gated=True
)


if __name__ == "__main__":
    common = dict(
        e=args.num_experts,
        topk=args.topk,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
    )
    print(f"config: {common}")
    for bench in (benchmark_quant, benchmark_silu_mul):
        bench.run(print_data=True, save_path=args.save_path, **common)

    print("\nBenchmark finished!")
