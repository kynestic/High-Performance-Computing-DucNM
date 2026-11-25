"""Experiment 1: Prove that matrix multiplication (GEMM) is the dominant bottleneck in CNNs.

Evidence collected:
  1. Direct nested-loop convolution vs im2col + GEMM on CPU (why frameworks convert conv to GEMM).
  2. Exact analytical FLOP breakdown of our 54,314-param CNN across Conv1, Conv2, FC1, FC2 vs
     ReLU/MaxPool/Softmax/SGD.
  3. Real CPU wall-clock breakdown (im2col/col2im vs GEMM vs elementwise/pool/SGD) across batch sizes.
  4. Mock GPU kernel breakdown (explicitly labeled as simulated) calibrated to T4-class hardware
     (compute-bound GEMM vs bandwidth-bound elementwise kernels), matching the ~94-95% nsys share
     reported in ltnaphixay/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from cnn_core import SmallCNN, conv_forward_direct_naive, im2col, make_synthetic_images


def benchmark_direct_vs_im2col(batch: int = 4, repeats: int = 3) -> Dict:
    """Compare direct 7-nested-loop Conv2D vs im2col + GEMM on Conv1 (1->8, 5x5, 28x28)."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((batch, 1, 28, 28)).astype(np.float32)
    w_mat = rng.standard_normal((8, 25)).astype(np.float32)
    w_4d = w_mat.reshape(8, 1, 5, 5)
    b = rng.standard_normal(8).astype(np.float32)

    out_ref = conv_forward_direct_naive(x, w_4d, b, pad=2)
    cols = im2col(x, ksize=5, pad=2)
    out_gemm = (w_mat @ cols + b[:, None]).reshape(8, batch, 28, 28).transpose(1, 0, 2, 3)
    max_diff = float(np.max(np.abs(out_ref - out_gemm)))

    t_direct = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = conv_forward_direct_naive(x, w_4d, b, pad=2)
        t_direct.append((time.perf_counter() - t0) * 1000.0)

    t_im2col = []
    for _ in range(max(10, repeats * 5)):
        t0 = time.perf_counter()
        c = im2col(x, ksize=5, pad=2)
        _ = (w_mat @ c + b[:, None]).reshape(8, batch, 28, 28).transpose(1, 0, 2, 3)
        t_im2col.append((time.perf_counter() - t0) * 1000.0)

    med_direct = float(np.median(t_direct))
    med_gemm = float(np.median(t_im2col))
    return {
        "layer": "Conv1 (1->8, 5x5, 28x28)",
        "batch": batch,
        "max_abs_diff": max_diff,
        "direct_loop_ms": med_direct,
        "im2col_gemm_ms": med_gemm,
        "speedup_im2col_over_direct": med_direct / max(med_gemm, 1e-9),
        "source": "real_cpu_measurement",
    }


def benchmark_cnn_breakdown(batches: List[int], repeats: int = 10) -> List[Dict]:
    """Measure real CPU step breakdown and exact FLOP shares across batch sizes."""
    rows = []
    for b in batches:
        net = SmallCNN(seed=2026)
        x, y = make_synthetic_images(b, seed=100 + b)
        # Warm-up
        for _ in range(2):
            net.step(x, y, apply_update=True, profile=True)

        samples = []
        for _ in range(repeats):
            _, _, _, t = net.step(x, y, apply_update=True, profile=True)
            samples.append(t)

        med = {k: float(np.median([s[k] for s in samples])) * 1000.0 for k in samples[0]}
        flops = SmallCNN.step_flops(b)

        gemm_ms = med["total_gemm"]
        im2col_ms = med["total_im2col_col2im"]
        elem_ms = med["total_elementwise_pool_sgd"]
        # Sum of component medians so percentages sum cleanly to 100.0%
        total_ms = gemm_ms + im2col_ms + elem_ms

        rows.append({
            "batch": b,
            "flops": flops,
            "gemm_flop_pct": 100.0 * flops["total_gemm"] / flops["total_flops"],
            "conv_gemm_flop_pct": 100.0 * flops["conv_gemm"] / flops["total_flops"],
            "fc_gemm_flop_pct": 100.0 * flops["fc_gemm"] / flops["total_flops"],
            "cpu_total_ms": total_ms,
            "cpu_conv_gemm_ms": med["conv_gemm_fwd"] + med["conv_gemm_bwd"],
            "cpu_fc_gemm_ms": med["fc_gemm_fwd"] + med["fc_gemm_bwd"],
            "cpu_gemm_ms": gemm_ms,
            "cpu_im2col_col2im_ms": im2col_ms,
            "cpu_elementwise_pool_sgd_ms": elem_ms,
            "cpu_gemm_time_pct": 100.0 * gemm_ms / total_ms,
            "cpu_im2col_time_pct": 100.0 * im2col_ms / total_ms,
            "cpu_elem_time_pct": 100.0 * elem_ms / total_ms,
            "cpu_effective_gflops": (flops["total_flops"] / 1e9) / (total_ms / 1000.0),
            "source": "real_cpu_measurement",
        })
    return rows


def mock_gpu_kernel_breakdown(batches: List[int]) -> List[Dict]:
    """Simulate GPU kernel time breakdown for the same CNN using a roofline kernel model.

    Explicitly marked as simulated/mock output:
      - GEMM kernels are compute-bound on small matrices (~250-900 GFLOP/s effective on T4-class GPU
        plus ~12 us kernel launch overhead per GEMM call).
      - Elementwise / pooling / im2col / SGD kernels are memory-bandwidth-bound (~220 GB/s effective
        plus ~5 us launch overhead per kernel call), so they take a larger share of GPU time (~6-15%)
        than their share of FLOPs (~1.5%), yet GEMM still dominates GPU execution time (~85-94%).
    """
    rows = []
    for b in batches:
        flops = SmallCNN.step_flops(b)
        # Effective GFLOP/s scales up with batch (more parallelism/data reuse on GPU)
        eff_gflops = 180.0 + 720.0 * (b / (b + 128.0))
        gemm_compute_ms = (flops["total_gemm"] / (eff_gflops * 1e9)) * 1000.0
        gemm_launch_ms = 11 * 0.010  # 11 GEMM calls x 10 us launch
        gemm_ms = gemm_compute_ms + gemm_launch_ms

        # Bytes touched by im2col/col2im + ReLU/pool/softmax/SGD
        act_bytes = b * 4 * (28 * 28 * 8 * 4 + 14 * 14 * 16 * 4 + 64 * 4 + 10 * 2)
        param_bytes = 54314 * 4 * 3  # read/write weights+vel+grad
        mem_bw_bytes_per_s = 220e9
        non_gemm_ms = ((act_bytes + param_bytes) / mem_bw_bytes_per_s) * 1000.0 + 12 * 0.005

        total_ms = gemm_ms + non_gemm_ms
        rows.append({
            "batch": b,
            "gpu_profile": "Simulated Tesla T4 (compute-bound GEMM + BW-bound elementwise)",
            "mock_gemm_ms": round(gemm_ms, 4),
            "mock_non_gemm_ms": round(non_gemm_ms, 4),
            "mock_total_kernel_ms": round(total_ms, 4),
            "mock_gemm_time_pct": round(100.0 * gemm_ms / total_ms, 2),
            "mock_non_gemm_time_pct": round(100.0 * non_gemm_ms / total_ms, 2),
            "source": "mock_simulation",
        })
    return rows


def benchmark_wide_conv_layer(batch: int = 64, cin: int = 32, cout: int = 64, side: int = 28,
                              ksize: int = 3, repeats: int = 5) -> Dict:
    """Measure a realistic mid-network Conv layer (32->64 channels) fwd+bwd on CPU.

    Shows that once channel dimensions reach realistic CNN widths (e.g. 32->64),
    GEMM compute scales as O(cin * cout) while im2col/col2im/ReLU scale only as O(cin + cout),
    making GEMM dominate real CPU wall-clock time as well as FLOPs.
    """
    from cnn_core import col2im
    rng = np.random.default_rng(99)
    pad = ksize // 2
    x = rng.standard_normal((batch, cin, side, side)).astype(np.float32)
    w = rng.standard_normal((cout, cin * ksize * ksize)).astype(np.float32)
    b = rng.standard_normal(cout).astype(np.float32)
    dz = rng.standard_normal((batch, cout, side, side)).astype(np.float32)

    # Warm-up
    c0 = im2col(x, ksize=ksize, pad=pad)
    _ = w @ c0 + b[:, None]

    t_im2col, t_gemm, t_elem = [], [], []
    for _ in range(repeats):
        s0 = time.perf_counter()
        cols = im2col(x, ksize=ksize, pad=pad)
        s1 = time.perf_counter()
        z = w @ cols
        s2 = time.perf_counter()
        z += b[:, None]
        _ = np.maximum(z, 0.0)
        dz_mat = dz.transpose(1, 0, 2, 3).reshape(cout, batch * side * side)
        s3 = time.perf_counter()
        _ = dz_mat @ cols.T
        dcols = w.T @ dz_mat
        s4 = time.perf_counter()
        _ = col2im(dcols, x.shape, ksize=ksize, pad=pad)
        s5 = time.perf_counter()

        t_im2col.append(((s1 - s0) + (s5 - s4)) * 1000.0)
        t_gemm.append(((s2 - s1) + (s4 - s3)) * 1000.0)
        t_elem.append((s3 - s2) * 1000.0)

    gemm_ms = float(np.median(t_gemm))
    im2col_ms = float(np.median(t_im2col))
    elem_ms = float(np.median(t_elem))
    total_ms = gemm_ms + im2col_ms + elem_ms
    # 3 GEMMs (fwd, dW, dX): each 2 * cout * (cin*k*k) * (batch*side*side)
    gemm_flops = 3 * 2 * cout * (cin * ksize * ksize) * (batch * side * side)
    elem_flops = 2 * cout * batch * side * side
    return {
        "layer": f"Conv {cin}->{cout} ({ksize}x{ksize}, {side}x{side}, batch={batch})",
        "gemm_flop_pct": 100.0 * gemm_flops / (gemm_flops + elem_flops),
        "cpu_total_ms": total_ms,
        "cpu_gemm_ms": gemm_ms,
        "cpu_im2col_col2im_ms": im2col_ms,
        "cpu_elem_ms": elem_ms,
        "cpu_gemm_time_pct": 100.0 * gemm_ms / total_ms,
        "cpu_im2col_time_pct": 100.0 * im2col_ms / total_ms,
        "cpu_elem_time_pct": 100.0 * elem_ms / total_ms,
        "source": "real_cpu_measurement",
    }


def run(batches: List[int] = None, repeats: int = 10) -> Dict:
    if batches is None:
        batches = [16, 64, 128, 256]
    direct_cmp = benchmark_direct_vs_im2col(batch=4, repeats=3)
    cpu_breakdown = benchmark_cnn_breakdown(batches=batches, repeats=repeats)
    wide_conv = benchmark_wide_conv_layer(batch=64, cin=32, cout=64, side=28, ksize=3, repeats=5)
    gpu_mock = mock_gpu_kernel_breakdown(batches=batches)
    return {
        "experiment": "exp1_cnn_matmul_bottleneck",
        "model": "SmallCNN (1x28x28 -> Conv8@5x5 -> Pool -> Conv16@5x5 -> Pool -> FC64 -> FC10)",
        "parameters": 54314,
        "direct_vs_im2col": direct_cmp,
        "cpu_breakdown": cpu_breakdown,
        "wide_conv_cpu": wide_conv,
        "gpu_kernel_breakdown_mock": gpu_mock,
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 1: Matrix multiplication bottleneck in CNN")
    parser.add_argument("--batches", default="16,64,128,256")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", default="outputs/exp1_cnn_bottleneck.json")
    args = parser.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    res = run(batches=batches, repeats=args.repeats)

    out_path = Path(__file__).resolve().parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    dc = res["direct_vs_im2col"]
    print("=== EXP 1: CNN MATRIX MULTIPLICATION BOTTLENECK ===")
    print(f"Direct loop vs im2col+GEMM (batch={dc['batch']}): "
          f"{dc['direct_loop_ms']:.2f} ms -> {dc['im2col_gemm_ms']:.2f} ms "
          f"({dc['speedup_im2col_over_direct']:.1f}x faster, diff={dc['max_abs_diff']:.2e})")
    print("\nCPU Real Measurements (FLOPs & Wall-clock Breakdown):")
    print(f"{'Batch':>6} | {'GEMM FLOP%':>10} | {'Total(ms)':>9} | {'GEMM(ms)':>8} | {'GEMM%':>7} | {'im2col%':>7} | {'Elem%':>6}")
    for r in res["cpu_breakdown"]:
        print(f"{r['batch']:>6} | {r['gemm_flop_pct']:>9.2f}% | {r['cpu_total_ms']:>9.2f} | "
              f"{r['cpu_gemm_ms']:>8.2f} | {r['cpu_gemm_time_pct']:>6.1f}% | "
              f"{r['cpu_im2col_time_pct']:>6.1f}% | {r['cpu_elem_time_pct']:>5.1f}%")
    wc = res["wide_conv_cpu"]
    print(f"\nWide Layer ({wc['layer']}): GEMM FLOP={wc['gemm_flop_pct']:.2f}%, "
          f"total={wc['cpu_total_ms']:.2f} ms -> GEMM={wc['cpu_gemm_time_pct']:.1f}%, "
          f"im2col={wc['cpu_im2col_time_pct']:.1f}%, elem={wc['cpu_elem_time_pct']:.1f}%")
    print("\nMock GPU Kernel Time Share (Simulated T4 Roofline):")
    print(f"{'Batch':>6} | {'Kernel(ms)':>10} | {'GEMM(ms)':>9} | {'GEMM%':>7} | {'Non-GEMM%':>9}")
    for r in res["gpu_kernel_breakdown_mock"]:
        print(f"{r['batch']:>6} | {r['mock_total_kernel_ms']:>10.4f} | {r['mock_gemm_ms']:>9.4f} | "
              f"{r['mock_gemm_time_pct']:>6.1f}% | {r['mock_non_gemm_time_pct']:>8.1f}%")


if __name__ == "__main__":
    main()
