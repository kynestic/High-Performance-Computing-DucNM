"""Small end-to-end benchmark for CPU/GPU training and distributed communication cost.

The problem is ridge linear regression:
    y = X @ w_true + noise
    min_w  0.5 * mean((X @ w - y)^2) + 0.5 * lambda * mean(w^2)

This intentionally uses matrix multiplication for the forward and gradient paths.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def set_seed(seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    return rng


def make_data(n, d, noise, rng):
    X = rng.standard_normal((n, d), dtype=np.float32)
    w_true = rng.standard_normal(d, dtype=np.float32)
    w_true /= max(1.0, np.linalg.norm(w_true))
    y = X @ w_true + rng.normal(0.0, noise, n).astype(np.float32)
    return X, y, w_true


def cpu_train(X, y, lr, epochs, batch, lam):
    n, d = X.shape
    rng = np.random.default_rng(123)
    w = rng.standard_normal(d, dtype=np.float32) * 0.01
    total_seconds = 0.0

    for epoch in range(epochs):
        perm = rng.permutation(n)
        begin = time.perf_counter()
        for at in range(0, n, batch):
            idx = perm[at:at + batch]
            xb = X[idx]
            yb = y[idx]
            pred = xb @ w                         # forward GEMM/GEMV
            residual = pred - yb
            grad = (xb.T @ residual) / len(idx)  # gradient GEMM
            grad += lam * w
            w -= lr * grad
        total_seconds += time.perf_counter() - begin

    pred = X @ w
    loss = float(np.mean((pred - y) ** 2) + lam * np.mean(w * w))
    return w, loss, total_seconds


def cpu_gemm(n, d, repeats):
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n, d), dtype=np.float32)
    B = rng.standard_normal((d, d), dtype=np.float32)
    for _ in range(3):
        _ = A @ B
    times = []
    for _ in range(repeats):
        begin = time.perf_counter()
        C = A @ B
        times.append(time.perf_counter() - begin)
        del C
    median = float(np.median(times))
    flops = 2.0 * n * d * d
    return median, flops / median / 1e9


def gpu_gemm(n, d, repeats):
    try:
        import torch
    except ImportError:
        return None, "PyTorch unavailable"
    if not torch.cuda.is_available():
        return None, "No CUDA device"

    torch.manual_seed(7)
    A = torch.randn((n, d), dtype=torch.float32, device="cuda")
    B = torch.randn((d, d), dtype=torch.float32, device="cuda")
    for _ in range(3):
        _ = A @ B
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        begin = time.perf_counter()
        C = A @ B
        torch.cuda.synchronize()
        times.append(time.perf_counter() - begin)
        del C
    median = float(np.median(times))
    flops = 2.0 * n * d * d
    return dict(median_seconds=median, gflops=flops / median / 1e9,
                device=torch.cuda.get_device_name(0)), None


def ring_allreduce_seconds(parameters, workers, alpha, bandwidth, repeats=100):
    """Ring reduce-scatter/allgather model: 2*(P-1)/P * message/ bandwidth + 2*(P-1)*alpha."""
    message_bytes = parameters * 4
    if workers <= 1:
        return 0.0
    return 2 * (workers - 1) / workers * message_bytes / bandwidth + 2 * (workers - 1) * alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lambda", dest="lam", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu-repeats", type=int, default=10)
    parser.add_argument("--alpha-us", type=float, default=3.0)
    parser.add_argument("--bandwidth-gbps", type=float, default=40.0)
    args = parser.parse_args()

    rng = set_seed(args.seed)
    X, y, w_true = make_data(args.n, args.d, 0.05, rng)
    w, loss, cpu_time = cpu_train(X, y, args.lr, args.epochs, args.batch, args.lam)

    gemm_n = min(args.n, 4096)
    gemm_d = min(args.d, 512)
    cpu_gemm_seconds, cpu_gemm_gflops = cpu_gemm(gemm_n, gemm_d, args.gpu_repeats)
    gpu, gpu_reason = gpu_gemm(gemm_n, gemm_d, args.gpu_repeats)

    alpha = args.alpha_us * 1e-6
    bandwidth = args.bandwidth_gbps * 1e9 / 8
    comm = []
    batches_per_epoch = math.ceil(args.n / args.batch)
    for workers in (2, 4, 8):
        step_comm = ring_allreduce_seconds(args.d, workers, alpha, bandwidth)
        epoch_comm = step_comm * batches_per_epoch
        comm.append({
            "workers": workers,
            "gradient_bytes": args.d * 4,
            "step_comm_seconds": step_comm,
            "epoch_comm_seconds": epoch_comm,
            "epochs_comm_seconds": epoch_comm * args.epochs,
        })

    result = {
        "problem": "ridge linear regression",
        "data": {"n": args.n, "d": args.d, "batch": args.batch, "epochs": args.epochs,
                 "lr": args.lr, "lambda": args.lam, "seed": args.seed},
        "training": {"cpu_seconds": cpu_time, "loss": loss,
                     "cpu_weight_error": float(np.linalg.norm(w - w_true) / max(1.0, np.linalg.norm(w_true)))},
        "gemm": {"shape": [gemm_n, gemm_d, gemm_d],
                 "cpu": {"median_seconds": cpu_gemm_seconds, "gflops": cpu_gemm_gflops},
                 "gpu": gpu, "gpu_reason": gpu_reason},
        "communication": {"model": "ring allreduce reduce-scatter + allgather",
                          "alpha_seconds": alpha, "bandwidth_bytes_per_second": bandwidth,
                          "batches_per_epoch": batches_per_epoch, "rows": comm},
    }

    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_demo.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
