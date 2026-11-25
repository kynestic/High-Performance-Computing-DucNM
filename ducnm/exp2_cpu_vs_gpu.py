"""Experiment 2: Compare CPU vs GPU in CNN training and core layers (Conv & Matvec/GEMM).

Measures on local CPU (real NumPy):
  1. End-to-end 5-epoch CNN training on 2,048 synthetic 28x28 images (batch=64).
  2. Paper's classic Conv layer (224x224, 96 filters 11x11, stride 4) across batch=1..128.
  3. Matvec (O(n^2)) and GEMM (O(n^3)) across sizes n=256..4096.

GPU measurements:
  - Uses real PyTorch+CUDA automatically if installed and a CUDA GPU is available.
  - Otherwise generates explicitly labeled Mock GPU outputs using a transparent hardware model
    (launch latency + roofline compute/bandwidth + optional H2D/D2H PCIe transfer), calibrated
    to Tesla T4 / RTX-class specs, demonstrating BOTH where GPU dominates (large batch/matrix)
    and where GPU loses to CPU (batch=1 or tiny n=256 due to kernel launch & PCIe overhead).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cnn_core import SmallCNN, make_synthetic_images


def cpu_train_cnn(n_train: int = 2048, n_val: int = 512, batch: int = 64, epochs: int = 5,
                  lr: float = 0.05, momentum: float = 0.9, seed: int = 2026) -> Dict:
    """Train SmallCNN on synthetic 28x28 images on CPU and record per-epoch metrics."""
    x_tr, y_tr = make_synthetic_images(n_train, seed=seed)
    x_va, y_va = make_synthetic_images(n_val, seed=seed + 1)
    net = SmallCNN(seed=seed)
    rng = np.random.default_rng(seed + 2)

    history = []
    total_train_s = 0.0
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n_train)
        t0 = time.perf_counter()
        ep_loss = 0.0
        ep_acc = 0.0
        steps = 0
        for at in range(0, n_train, batch):
            idx = perm[at : at + batch]
            loss, acc, _, _ = net.step(x_tr[idx], y_tr[idx], lr=lr, momentum=momentum, apply_update=True)
            ep_loss += loss
            ep_acc += acc
            steps += 1
        dt = time.perf_counter() - t0
        total_train_s += dt

        val_logits, _, _ = net.forward_logits(x_va)
        val_acc = float((np.argmax(val_logits, axis=1) == y_va).mean())
        history.append({
            "epoch": ep,
            "train_loss": ep_loss / steps,
            "train_acc": ep_acc / steps,
            "val_acc": val_acc,
            "epoch_seconds": dt,
        })

    return {
        "device": "CPU (NumPy)",
        "n_train": n_train,
        "n_val": n_val,
        "batch": batch,
        "epochs": epochs,
        "total_seconds": total_train_s,
        "avg_epoch_seconds": total_train_s / epochs,
        "final_val_acc": history[-1]["val_acc"],
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
        "source": "real_cpu_measurement",
    }


def _im2col_paper_layer(x: np.ndarray, k: int = 11, stride: int = 4) -> np.ndarray:
    """im2col for paper's AlexNet-like first conv layer: (N, 1, 224, 224), 11x11, stride 4 -> out 54x54."""
    n, c, h, w = x.shape
    oh = (h - k) // stride + 1
    ow = (w - k) // stride + 1
    cols = np.empty((c * k * k, n * oh * ow), dtype=x.dtype)
    idx = 0
    for ci in range(c):
        for ky in range(k):
            for kx in range(k):
                patch = x[:, ci, ky : ky + oh * stride : stride, kx : kx + ow * stride : stride]
                cols[idx] = patch.reshape(-1)
                idx += 1
    return cols


def bench_cpu_conv_paper(batches: List[int], repeats: int = 5) -> List[Dict]:
    """Measure CPU forward time of 224x224 -> 96@11x11/s4 conv via im2col + GEMM."""
    rng = np.random.default_rng(12)
    w_mat = rng.standard_normal((96, 121)).astype(np.float32)
    rows = []
    for b in batches:
        x = rng.standard_normal((b, 1, 224, 224)).astype(np.float32)
        _ = w_mat @ _im2col_paper_layer(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            cols = _im2col_paper_layer(x)
            _ = w_mat @ cols
            times.append((time.perf_counter() - t0) * 1000.0)
        med_ms = float(np.median(times))
        flops = 2.0 * b * 96 * 121 * 2916
        rows.append({
            "batch": b,
            "cpu_ms": med_ms,
            "cpu_ms_per_img": med_ms / b,
            "cpu_gflops": (flops / 1e9) / (med_ms / 1000.0),
            "flops": flops,
            "source": "real_cpu_measurement",
        })
    return rows


def bench_cpu_fc(matvec_sizes: List[int], gemm_sizes: List[int], repeats: int = 7) -> Dict:
    """Measure CPU Matvec (O(n^2)) and GEMM (O(n^3))."""
    rng = np.random.default_rng(13)
    mv_rows = []
    for n in matvec_sizes:
        a = rng.standard_normal((n, n)).astype(np.float32)
        x = rng.standard_normal(n).astype(np.float32)
        _ = a @ x
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = a @ x
            times.append((time.perf_counter() - t0) * 1000.0)
        med_ms = float(np.median(times))
        nbytes = 4.0 * (n * n + 2 * n)
        mv_rows.append({
            "n": n,
            "cpu_ms": med_ms,
            "cpu_gbps": (nbytes / 1e9) / (med_ms / 1000.0),
            "source": "real_cpu_measurement",
        })

    gm_rows = []
    for n in gemm_sizes:
        a = rng.standard_normal((n, n)).astype(np.float32)
        b = rng.standard_normal((n, n)).astype(np.float32)
        _ = a @ b
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = a @ b
            times.append((time.perf_counter() - t0) * 1000.0)
        med_ms = float(np.median(times))
        flops = 2.0 * (n ** 3)
        gm_rows.append({
            "n": n,
            "cpu_ms": med_ms,
            "cpu_gflops": (flops / 1e9) / (med_ms / 1000.0),
            "source": "real_cpu_measurement",
        })

    return {"matvec": mv_rows, "gemm": gm_rows}


def try_real_gpu_bench(conv_batches: List[int], gemm_sizes: List[int]) -> Tuple[Optional[Dict], str]:
    """Attempt real GPU benchmarks with PyTorch CUDA if installed."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return None, "PyTorch unavailable in current Python environment"
    if not torch.cuda.is_available():
        return None, "CUDA GPU unavailable to PyTorch"

    dev = torch.device("cuda")
    w = torch.randn(96, 1, 11, 11, device=dev)
    b = torch.randn(96, device=dev)
    conv_rows = []
    for n in conv_batches:
        x = torch.randn(n, 1, 224, 224, device=dev)
        for _ in range(3):
            _ = F.conv2d(x, w, b, stride=4)
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = F.conv2d(x, w, b, stride=4)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        conv_rows.append({"batch": n, "gpu_ms": float(np.median(ts)), "source": "real_gpu_measurement"})

    gemm_rows = []
    for n in gemm_sizes:
        a = torch.randn(n, n, device=dev)
        c = torch.randn(n, n, device=dev)
        for _ in range(3):
            _ = a @ c
        torch.cuda.synchronize()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = a @ c
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        gemm_rows.append({"n": n, "gpu_ms": float(np.median(ts)), "source": "real_gpu_measurement"})

    return {
        "device": torch.cuda.get_device_name(0),
        "conv": conv_rows,
        "gemm": gemm_rows,
    }, ""


def mock_gpu_comparison(cpu_train: Dict, cpu_conv: List[Dict], cpu_fc: Dict) -> Dict:
    """Generate transparent, hardware-grounded Mock GPU results (Tesla T4 profile: 8.1 TFLOP/s FP32, 320 GB/s).

    Explicitly models:
      - Kernel launch latency (~25 us per kernel / ~220 us per training step)
      - Effective GPU utilization scaling with batch/matrix size
      - Optional PCIe Gen3 H2D/D2H host-device transfer overhead (~10 GB/s)
    """
    # 1. End-to-end 5-epoch CNN training (SmallCNN, 2048 train, batch=64 -> 32 steps/epoch)
    steps_per_epoch = cpu_train["n_train"] // cpu_train["batch"]
    flops_per_step = SmallCNN.step_flops(cpu_train["batch"])["total_flops"]
    # Small batch=64 CNN on GPU has ~220 us launch overhead + ~150 us kernel compute (~0.37 ms/step)
    mock_step_compute_ms = (flops_per_step / (450e9)) * 1000.0 + 0.220
    mock_h2d_ms = (cpu_train["batch"] * 28 * 28 * 4 / 10e9) * 1000.0 + 0.025
    mock_epoch_kernel_s = steps_per_epoch * mock_step_compute_ms / 1000.0
    mock_epoch_with_pcie_s = steps_per_epoch * (mock_step_compute_ms + mock_h2d_ms) / 1000.0

    train_mock = {
        "device": "Simulated NVIDIA Tesla T4 (mock)",
        "epochs": cpu_train["epochs"],
        "avg_epoch_seconds_kernel_only": round(mock_epoch_kernel_s, 5),
        "avg_epoch_seconds_with_h2d": round(mock_epoch_with_pcie_s, 5),
        "total_seconds_kernel_only": round(mock_epoch_kernel_s * cpu_train["epochs"], 4),
        "total_seconds_with_h2d": round(mock_epoch_with_pcie_s * cpu_train["epochs"], 4),
        "speedup_vs_cpu_kernel_only": round(cpu_train["avg_epoch_seconds"] / mock_epoch_kernel_s, 2),
        "speedup_vs_cpu_with_h2d": round(cpu_train["avg_epoch_seconds"] / mock_epoch_with_pcie_s, 2),
        "final_val_acc": round(cpu_train["final_val_acc"], 4),  # Same math -> same accuracy
        "source": "mock_simulation",
    }

    # 2. Conv paper layer (224x224, 96@11x11/s4): t_gpu(B) = a_launch + b_per_img * B
    # From hieunv/ analysis: ~25 us launch + ~0.068 ms/img on T4 (im2col+GEMM)
    conv_rows = []
    for r in cpu_conv:
        b = r["batch"]
        gpu_ms = 0.025 + 0.068 * b
        pcie_ms = (b * (224 * 224 + 96 * 54 * 54) * 4 / 10e9) * 1000.0 + 0.030
        conv_rows.append({
            "batch": b,
            "cpu_ms": round(r["cpu_ms"], 3),
            "mock_gpu_kernel_ms": round(gpu_ms, 4),
            "mock_gpu_with_pcie_ms": round(gpu_ms + pcie_ms, 4),
            "speedup_kernel_only": round(r["cpu_ms"] / gpu_ms, 2),
            "speedup_with_pcie": round(r["cpu_ms"] / (gpu_ms + pcie_ms), 2),
            "source": "mock_simulation",
        })

    # 3. Matvec (bandwidth-bound: 267 GB/s peak + 25 us launch) & GEMM (compute-bound: ~3.5 TFLOP/s + 25 us)
    mv_rows = []
    for r in cpu_fc["matvec"]:
        n = r["n"]
        nbytes = 4.0 * (n * n + 2 * n)
        gpu_ms = 0.025 + (nbytes / 267e9) * 1000.0
        mv_rows.append({
            "n": n,
            "cpu_ms": round(r["cpu_ms"], 4),
            "mock_gpu_ms": round(gpu_ms, 4),
            "speedup": round(r["cpu_ms"] / gpu_ms, 2),
            "note": "GPU slower at tiny n due to 25us kernel launch overhead" if r["cpu_ms"] < gpu_ms else "GPU BW wins",
            "source": "mock_simulation",
        })

    gm_rows = []
    for r in cpu_fc["gemm"]:
        n = r["n"]
        flops = 2.0 * (n ** 3)
        eff_tflops = 3.5e12 * (n / (n + 512.0))
        gpu_ms = 0.025 + (flops / eff_tflops) * 1000.0
        gm_rows.append({
            "n": n,
            "cpu_ms": round(r["cpu_ms"], 4),
            "mock_gpu_ms": round(gpu_ms, 4),
            "speedup": round(r["cpu_ms"] / gpu_ms, 2),
            "source": "mock_simulation",
        })

    return {
        "training_mock": train_mock,
        "conv_mock": conv_rows,
        "matvec_mock": mv_rows,
        "gemm_mock": gm_rows,
    }


def run(epochs: int = 5) -> Dict:
    conv_batches = [1, 8, 16, 32, 64, 128]
    mv_sizes = [256, 512, 1024, 2048, 4096]
    gm_sizes = [256, 512, 1024, 2048]

    train_cpu = cpu_train_cnn(n_train=2048, n_val=512, batch=64, epochs=epochs)
    conv_cpu = bench_cpu_conv_paper(conv_batches, repeats=4)
    fc_cpu = bench_cpu_fc(mv_sizes, gm_sizes, repeats=6)
    real_gpu, gpu_reason = try_real_gpu_bench(conv_batches, gm_sizes)
    mock_gpu = mock_gpu_comparison(train_cpu, conv_cpu, fc_cpu)

    return {
        "experiment": "exp2_cpu_vs_gpu",
        "cpu_training": train_cpu,
        "cpu_conv_paper": conv_cpu,
        "cpu_fc": fc_cpu,
        "real_gpu": real_gpu,
        "real_gpu_reason": gpu_reason if real_gpu is None else "CUDA available",
        "mock_gpu": mock_gpu,
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 2: CPU vs GPU in CNN training and layers")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--out", default="outputs/exp2_cpu_vs_gpu.json")
    args = parser.parse_args()

    res = run(epochs=args.epochs)
    out_path = Path(__file__).resolve().parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    tr = res["cpu_training"]
    tm = res["mock_gpu"]["training_mock"]
    print("=== EXP 2: GPU VS CPU IN TRAINING ===")
    print(f"Real CPU CNN Training (2048 imgs, batch=64, {tr['epochs']} epochs): "
          f"{tr['total_seconds']:.3f}s ({tr['avg_epoch_seconds']*1000:.1f} ms/epoch), "
          f"loss={tr['final_train_loss']:.4f}, val_acc={tr['final_val_acc']*100:.1f}%")
    print(f"Real GPU status: {res['real_gpu_reason']}")
    print(f"Mock GPU Training (T4 model): {tm['total_seconds_kernel_only']:.4f}s kernel-only "
          f"({tm['speedup_vs_cpu_kernel_only']:.1f}x vs CPU), "
          f"{tm['total_seconds_with_h2d']:.4f}s incl. H2D PCIe ({tm['speedup_vs_cpu_with_h2d']:.1f}x vs CPU)")

    print("\nConv Layer (224x224, 96@11x11, s=4) CPU Real vs GPU Mock:")
    print(f"{'Batch':>6} | {'CPU(ms)':>9} | {'GPU_kern(ms)':>12} | {'Speedup':>8} | {'GPU+PCIe(ms)':>12} | {'Speedup+PCIe':>12}")
    for r in res["mock_gpu"]["conv_mock"]:
        print(f"{r['batch']:>6} | {r['cpu_ms']:>9.2f} | {r['mock_gpu_kernel_ms']:>12.4f} | "
              f"{r['speedup_kernel_only']:>7.1f}x | {r['mock_gpu_with_pcie_ms']:>12.4f} | {r['speedup_with_pcie']:>11.1f}x")

    print("\nMatvec CPU Real vs GPU Mock (shows small-n overhead inversion):")
    print(f"{'n':>6} | {'CPU(ms)':>9} | {'GPU(ms)':>9} | {'Speedup':>8} | Note")
    for r in res["mock_gpu"]["matvec_mock"]:
        print(f"{r['n']:>6} | {r['cpu_ms']:>9.4f} | {r['mock_gpu_ms']:>9.4f} | {r['speedup']:>7.2f}x | {r['note']}")


if __name__ == "__main__":
    main()
