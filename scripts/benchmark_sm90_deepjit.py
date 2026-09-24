#!/usr/bin/env python3
"""Minimal SM90 benchmark for DeepGEMM kernels that use DeepJIT.

Run:
  python scripts/benchmark_sm90_deepjit.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
PYTHON = os.environ.get("PYTHON", sys.executable)

CASES = (
    "bf16_gemm_nt",
    "fp8_gemm_nt",
    "grouped_fp8_gemm_nt",
    "tf32_hc_prenorm_gemm",
    "mqa_logits",
)

WARMUP_ITERS = 10
MEASURE_ITERS = 500
TRIALS = 5
SEED = 0


def build_case(name: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))

    import torch
    import deep_gemm
    from deep_gemm.testing.numeric import count_bytes
    from deep_gemm.utils import per_custom_dims_cast_to_fp8
    from generators import (
        KernelType,
        MajorTypeAB,
        generate_m_grouped_contiguous,
        generate_normal,
    )

    if name == "bf16_gemm_nt":
        m, n, k = 4096, 7168, 2048
        a, b, _, d, _ = generate_normal(
            m=m,
            n=n,
            k=k,
            major_a=MajorTypeAB.KMajor,
            major_b=MajorTypeAB.KMajor,
            accumulate=False,
            out_dtype=torch.bfloat16,
            kernel_type=KernelType.KernelNoSF,
            use_bf16=True,
        )
        return {
            "description": f"bf16_gemm_nt  m={m} n={n} k={k}",
            "run": lambda: deep_gemm.bf16_gemm_nt(a, b, d),
            "flops": float(2 * m * n * k),
            "bytes": float(count_bytes(a, b, d)),
        }

    if name == "fp8_gemm_nt":
        m, n, k = 4096, 7168, 2048
        a, b, _, d, _ = generate_normal(
            m=m,
            n=n,
            k=k,
            major_a=MajorTypeAB.KMajor,
            major_b=MajorTypeAB.KMajor,
            accumulate=False,
            out_dtype=torch.bfloat16,
            kernel_type=KernelType.Kernel1D2D,
            use_ue8m0=False,
        )
        return {
            "description": f"fp8_gemm_nt   m={m} n={n} k={k}",
            "run": lambda: deep_gemm.fp8_gemm_nt(a, b, d, disable_ue8m0_cast=True),
            "flops": float(2 * m * n * k),
            "bytes": float(count_bytes(a, b, d)),
        }

    if name == "grouped_fp8_gemm_nt":
        groups, expected_m_per_group, n, k = 4, 1024, 4096, 2048
        deep_gemm.set_mk_alignment_for_contiguous_layout(
            deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        )
        m, a, b, grouped_layout, d, _, _ = generate_m_grouped_contiguous(
            num_groups=groups,
            expected_m_per_group=expected_m_per_group,
            n=n,
            k=k,
            major_a=MajorTypeAB.KMajor,
            major_b=MajorTypeAB.KMajor,
            use_ue8m0=False,
            use_psum_layout=False,
        )
        return {
            "description": f"grouped_fp8_nt groups={groups} m={m} n={n} k={k}",
            "run": lambda: deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a,
                b,
                d,
                grouped_layout,
                disable_ue8m0_cast=True,
                ensure_zero_padding=False,
            ),
            "flops": float(2 * m * n * k),
            "bytes": float(count_bytes(a, b, d, grouped_layout)),
        }

    if name == "tf32_hc_prenorm_gemm":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        m, n, k, splits = 4096, 24, 7168, 16
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((n, k), dtype=torch.float32, device="cuda")
        d = torch.empty((splits, m, n), dtype=torch.float32, device="cuda")
        sqr_sum = torch.empty((splits, m), dtype=torch.float32, device="cuda")
        return {
            "description": f"tf32_hc_prenorm_gemm m={m} n={n} k={k} splits={splits}",
            "run": lambda: deep_gemm.tf32_hc_prenorm_gemm(
                a, b, d, sqr_sum, num_splits=splits
            ),
            "flops": float(2 * m * n * k),
            "bytes": float(count_bytes(a, b, d, sqr_sum)),
        }

    if name == "mqa_logits":
        seq_len, seq_len_kv, heads, dim = 512, 4096, 32, 128
        q = torch.randn((seq_len, heads, dim), dtype=torch.bfloat16, device="cuda")
        kv = torch.randn((seq_len_kv, dim), dtype=torch.bfloat16, device="cuda")
        weights = torch.randn((seq_len, heads), dtype=torch.float32, device="cuda")
        q_in = (q.to(torch.float8_e4m3fn), None)
        kv_in = per_custom_dims_cast_to_fp8(kv, (0,), False)
        ks = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
        ke = torch.arange(seq_len, dtype=torch.int32, device="cuda") + (seq_len_kv - seq_len)

        element_size = torch.empty((), dtype=torch.float32).element_size()
        stride = ((weights.size(1) * element_size + 15) // 16) * 16 // element_size
        packed_weights = torch.empty(
            (weights.size(0), stride), dtype=torch.float32, device="cuda"
        )
        packed_weights[:, : weights.size(1)].copy_(weights)
        active_k = int((ke - ks).sum().item())

        return {
            "description": f"fp8_mqa_logits seq={seq_len} kv={seq_len_kv} h={heads} d={dim}",
            "run": lambda: deep_gemm.fp8_fp4_mqa_logits(
                q=q_in,
                kv=kv_in,
                weights=packed_weights,
                cu_seq_len_k_start=ks,
                cu_seq_len_k_end=ke,
                clean_logits=False,
            ),
            "flops": float(2 * active_k * heads * dim),
            "bytes": float(count_bytes(q_in, kv_in, packed_weights, ks, ke)),
        }

    raise ValueError(f"unknown case: {name}")


def run_child() -> int:
    import torch
    from deep_gemm.testing import get_arch_major

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if get_arch_major() != 9:
        raise RuntimeError(f"expected SM90, got arch major {get_arch_major()}")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    case = build_case(os.environ["DG_BENCH_CASE"])
    run = case["run"]

    torch.cuda.synchronize()
    start = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - start

    for _ in range(WARMUP_ITERS):
        run()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(MEASURE_ITERS):
        run()
    end_event.record()
    torch.cuda.synchronize()
    warm_s = start_event.elapsed_time(end_event) / MEASURE_ITERS / 1e3

    print(json.dumps({
        "description": case["description"],
        "first_call_s": first_call_s,
        "warm_s": warm_s,
        "tflops": case["flops"] / warm_s / 1e12 if warm_s > 0 else 0.0,
        "gbps": case["bytes"] / warm_s / 1e9 if warm_s > 0 else 0.0,
    }))
    return 0


def run_case(name: str, cache_dir: Path) -> dict:
    env = os.environ.copy()
    env["DG_BENCH_CASE"] = name
    env["DG_JIT_CACHE_DIR"] = str(cache_dir)
    result = subprocess.run(
        [PYTHON, str(Path(__file__).resolve())],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    return f"{seconds * 1e3:8.3f} ms"


def main() -> int:
    if "DG_BENCH_CASE" in os.environ:
        return run_child()

    with tempfile.TemporaryDirectory(prefix="deepjit_sm90_", dir="/tmp") as root:
        print(f"Using {PYTHON}")
        print()
        for name in CASES:
            for trial in range(1, TRIALS + 1):
                cache_dir = Path(root) / name / f"trial_{trial}"
                cold = run_case(name, cache_dir)
                cached = run_case(name, cache_dir)

                if cold["description"] != cached["description"]:
                    raise RuntimeError(
                        "cold and disk-cache processes generated different cases: "
                        f"{cold['description']!r} != {cached['description']!r}"
                    )

                print(f"{cold['description']} (trial {trial}/{TRIALS})")
                print(f"  cold first call:       {fmt_time(cold['first_call_s'])}")
                print(f"  disk-cache first call: {fmt_time(cached['first_call_s'])}")
                print(f"  warm call:             {fmt_time(cached['warm_s'])}")
                print(f"  throughput:            {cached['tflops']:8.2f} TFLOP/s")
                print(f"  bandwidth:             {cached['gbps']:8.2f} GB/s")
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
