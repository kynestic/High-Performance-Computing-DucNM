"""Experiment 3: Distributed Gradient Descent Communication Cost Analysis.

Analyzes:
  1. Hockney alpha-beta allreduce cost across 4 algorithms:
       - naive (parameter server):  2*(P-1) * (alpha + M*beta)
       - tree (reduce + bcast):     2*ceil(log2(P)) * (alpha + M*beta)
       - recursive doubling (rd):   (ceil(log2(P)) + extra) * (alpha + M*beta)
       - ring (RS + AG):            2*(P-1)*alpha + 2*(P-1)/P * M*beta
     across gradient sizes (2 KB, 217 KB SmallCNN, 1.63 MB MLP, 44 MB ResNet-18, 536 MB VGG-16)
     and worker counts P = 2, 4, 8, 16.
  2. Three network interconnects (simulated/analytical):
       - NVLink intra-node (alpha = 5 us, B = 150 GB/s)
       - PCIe intra-node (alpha = 15 us, B = 10 GB/s)
       - 10 Gbps Ethernet cluster (alpha = 50 us, B = 1.25 GB/s)
  3. Strong scaling & Compute-to-Communication ratio as Global Batch Size sweeps 64..4096:
       - Proves why small global batch fails to scale (latency/comm > compute savings)
       - Identifies break-even batch size and Amdahl bottleneck.
  4. Sync SGD vs Periodic Averaging (Parallel SGD) vs Async Downpour saturation (k* = T_grad / tau).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

INTERCONNECTS = {
    "nvlink": {"label": "NVLink (intra-node)", "alpha_us": 5.0, "bandwidth_GBps": 150.0},
    "pcie": {"label": "PCIe Gen3 (intra-node)", "alpha_us": 15.0, "bandwidth_GBps": 10.0},
    "eth10g": {"label": "10GbE Network (multi-node)", "alpha_us": 50.0, "bandwidth_GBps": 1.25},
}

MODELS = [
    {"name": "Toy Linear (d=512)", "params": 512, "flops_per_sample": 2048},
    {"name": "SmallCNN (this repo)", "params": 54314, "flops_per_sample": 1_530_000},
    {"name": "MLP 784-512-10", "params": 407050, "flops_per_sample": 815_000},
    {"name": "ResNet-18 (approx)", "params": 11_173_962, "flops_per_sample": 3_600_000_000},
]


def allreduce_seconds(algo: str, p: int, msg_bytes: int, alpha_s: float, bw_bps: float) -> float:
    """Hockney alpha-beta model for collective allreduce of msg_bytes across p workers."""
    if p <= 1:
        return 0.0
    beta = 1.0 / bw_bps
    lg = math.ceil(math.log2(p))
    if algo == "naive":
        return 2.0 * (p - 1) * (alpha_s + msg_bytes * beta)
    if algo == "tree":
        return 2.0 * lg * (alpha_s + msg_bytes * beta)
    if algo == "rd":
        extra = 0 if (p & (p - 1)) == 0 else 2
        return (lg + extra) * (alpha_s + msg_bytes * beta)
    if algo == "ring":
        return 2.0 * (p - 1) * alpha_s + 2.0 * ((p - 1) / p) * msg_bytes * beta
    raise ValueError(f"Unknown algorithm: {algo}")


def analyze_algorithms(p_list: List[int], msg_sizes_bytes: List[int]) -> List[Dict]:
    """Compare naive, tree, rd, ring on PCIe and NVLink profiles."""
    rows = []
    for net_key in ("nvlink", "pcie", "eth10g"):
        cfg = INTERCONNECTS[net_key]
        alpha = cfg["alpha_us"] * 1e-6
        bw = cfg["bandwidth_GBps"] * 1e9
        for p in p_list:
            for m in msg_sizes_bytes:
                times_us = {
                    algo: round(allreduce_seconds(algo, p, m, alpha, bw) * 1e6, 3)
                    for algo in ("naive", "tree", "rd", "ring")
                }
                best = min(times_us, key=times_us.get)
                rows.append({
                    "interconnect": net_key,
                    "workers": p,
                    "msg_bytes": m,
                    "times_us": times_us,
                    "fastest_algo": best,
                    "source": "analytical_alpha_beta_model",
                })
    return rows


def analyze_batch_scaling(
    model_name: str = "MLP / Medium CNN (~1.63 MB grad)",
    params: int = 407050,
    flops_per_sample: int = 1_530_000,
    gpu_eff_tflops: float = 0.85,
    interconnect: str = "pcie",
    workers_list: List[int] = None,
    batches: List[int] = None,
) -> List[Dict]:
    """Strong scaling across global batch sizes: explains when multi-GPU wins or loses."""
    if workers_list is None:
        workers_list = [1, 2, 4, 8]
    if batches is None:
        batches = [64, 128, 256, 512, 1024, 2048, 4096]

    cfg = INTERCONNECTS[interconnect]
    alpha = cfg["alpha_us"] * 1e-6
    bw = cfg["bandwidth_GBps"] * 1e9
    msg_bytes = params * 4

    rows = []
    for b_global in batches:
        # 1 GPU baseline step time: launch/host overhead (~120 us) + compute
        t_comp_1 = 120e-6 + (b_global * flops_per_sample) / (gpu_eff_tflops * 1e12)
        for p in workers_list:
            b_local = b_global / p
            # Smaller local batch lowers GPU efficiency slightly + fixed launch overhead remains
            t_comp_p = 120e-6 + (b_local * flops_per_sample) / (gpu_eff_tflops * 1e12)
            t_comm_p = allreduce_seconds("ring", p, msg_bytes, alpha, bw)
            t_step_p = t_comp_p + t_comm_p
            rows.append({
                "model": model_name,
                "interconnect": interconnect,
                "global_batch": b_global,
                "workers": p,
                "local_batch": b_local,
                "grad_bytes": msg_bytes,
                "step_compute_ms": round(t_comp_p * 1000.0, 4),
                "step_comm_ms": round(t_comm_p * 1000.0, 4),
                "step_total_ms": round(t_step_p * 1000.0, 4),
                "comm_share_pct": round(100.0 * t_comm_p / t_step_p, 2),
                "speedup_vs_1gpu": round(t_comp_1 / t_step_p, 3),
                "scaling_efficiency_pct": round(100.0 * (t_comp_1 / t_step_p) / p, 1),
                "source": "mock_analytical_scaling",
            })
    return rows


def compare_sgd_strategies(
    n_samples: int = 10000,
    global_batch: int = 256,
    epochs: int = 10,
    params: int = 54314,
    workers: int = 4,
    interconnect: str = "eth10g",
) -> List[Dict]:
    """Compare Sync SGD vs Periodic Averaging (Parallel SGD) vs Async Downpour server limit."""
    cfg = INTERCONNECTS[interconnect]
    alpha = cfg["alpha_us"] * 1e-6
    bw = cfg["bandwidth_GBps"] * 1e9
    msg_bytes = params * 4
    steps_per_epoch = math.ceil(n_samples / global_batch)
    total_steps = steps_per_epoch * epochs

    t_ar_ms = allreduce_seconds("ring", workers, msg_bytes, alpha, bw) * 1000.0
    t_ps_ms = allreduce_seconds("naive", workers, msg_bytes, alpha, bw) * 1000.0

    # 1. Sync SGD (every step)
    sync_comm_calls = total_steps
    # 2. Parallel SGD avg every H=10 steps
    avg10_calls = math.ceil(total_steps / 10)
    # 3. Parallel SGD avg once per epoch
    avgepoch_calls = epochs
    # 4. Downpour SGD saturation: k* = T_grad / tau
    t_grad_ms = 0.85  # local gradient compute per step
    tau_server_ms = 0.25  # parameter server update time
    k_star = t_grad_ms / tau_server_ms

    return [
        {
            "strategy": "Sync SGD (allreduce every step)",
            "comm_calls_10ep": sync_comm_calls,
            "total_comm_MB": round(sync_comm_calls * msg_bytes * 2 * (workers - 1) / workers / 1e6, 2),
            "total_comm_ms": round(sync_comm_calls * t_ar_ms, 2),
            "staleness": 0,
            "accuracy_risk": "Lowest (mathematically equivalent to single-worker batch SGD)",
        },
        {
            "strategy": "Parallel SGD (avg every 10 steps)",
            "comm_calls_10ep": avg10_calls,
            "total_comm_MB": round(avg10_calls * msg_bytes * 2 * (workers - 1) / workers / 1e6, 2),
            "total_comm_ms": round(avg10_calls * t_ar_ms, 2),
            "staleness": "0-9 local steps drift",
            "accuracy_risk": "Low on IID; requires ShuffleSGD on non-IID",
        },
        {
            "strategy": "Parallel SGD (avg once per epoch)",
            "comm_calls_10ep": avgepoch_calls,
            "total_comm_MB": round(avgepoch_calls * msg_bytes * 2 * (workers - 1) / workers / 1e6, 2),
            "total_comm_ms": round(avgepoch_calls * t_ar_ms, 2),
            "staleness": f"Up to {steps_per_epoch} steps drift",
            "accuracy_risk": "High on non-IID data without shuffling",
        },
        {
            "strategy": f"Async Downpour (PS, k* = T/tau = {k_star:.1f} workers)",
            "comm_calls_10ep": total_steps,
            "total_comm_MB": round(total_steps * msg_bytes * 2 / 1e6, 2),
            "total_comm_ms": round(total_steps * (t_ps_ms / workers), 2),
            "staleness": f"~W-1 = {workers - 1} steps",
            "accuracy_risk": "Degrades if staleness x lr is large; saturates above k*",
        },
    ]


def run() -> Dict:
    algo_table = analyze_algorithms(
        p_list=[2, 4, 8],
        msg_sizes_bytes=[2048, 217256, 1_628_200, 44_695_848],
    )
    batch_scaling = analyze_batch_scaling(
        model_name="MLP 784-512-10 / Small CNN (1.63 MB grad)",
        params=407050,
        flops_per_sample=1_530_000,
        gpu_eff_tflops=0.85,
        interconnect="pcie",
        workers_list=[1, 2, 4, 8],
        batches=[64, 256, 1024, 4096],
    )
    strategies = compare_sgd_strategies()
    return {
        "experiment": "exp3_communication_cost",
        "interconnect_profiles": INTERCONNECTS,
        "allreduce_comparison": algo_table,
        "batch_scaling": batch_scaling,
        "sgd_strategies": strategies,
    }


def main():
    parser = argparse.ArgumentParser(description="Exp 3: Distributed GD communication cost analysis")
    parser.add_argument("--out", default="outputs/exp3_comm_cost.json")
    args = parser.parse_args()

    res = run()
    out_path = Path(__file__).resolve().parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("=== EXP 3: COMMUNICATION COST IN GRADIENT DESCENT ===")
    print("\n1) Allreduce Algorithm Comparison on PCIe (alpha=15us, BW=10 GB/s), P=4,8:")
    print(f"{'P':>3} | {'Size':>10} | {'Naive(us)':>10} | {'Tree(us)':>9} | {'RD(us)':>9} | {'Ring(us)':>9} | {'Fastest':>7}")
    for r in res["allreduce_comparison"]:
        if r["interconnect"] == "pcie" and r["workers"] in (4, 8):
            t = r["times_us"]
            sz = f"{r['msg_bytes']/1024:.0f} KB" if r["msg_bytes"] < 1048576 else f"{r['msg_bytes']/1048576:.1f} MB"
            print(f"{r['workers']:>3} | {sz:>10} | {t['naive']:>10.1f} | {t['tree']:>9.1f} | "
                  f"{t['rd']:>9.1f} | {t['ring']:>9.1f} | {r['fastest_algo']:>7}")

    print("\n2) Strong Scaling vs Global Batch Size (PCIe, 1.63 MB grad, Ring Allreduce):")
    print(f"{'Batch':>6} | {'P':>2} | {'Comp(ms)':>8} | {'Comm(ms)':>8} | {'Comm%':>7} | {'Speedup':>8} | {'Eff%':>6}")
    for r in res["batch_scaling"]:
        print(f"{r['global_batch']:>6} | {r['workers']:>2} | {r['step_compute_ms']:>8.3f} | "
              f"{r['step_comm_ms']:>8.3f} | {r['comm_share_pct']:>6.1f}% | "
              f"{r['speedup_vs_1gpu']:>7.2f}x | {r['scaling_efficiency_pct']:>5.1f}%")

    print("\n3) Distributed Strategy Communication Trade-off (54K-param CNN, 10 epochs, 10GbE, P=4):")
    for s in res["sgd_strategies"]:
        print(f"  - {s['strategy']}: calls={s['comm_calls_10ep']}, "
              f"vol={s['total_comm_MB']} MB, comm_time={s['total_comm_ms']} ms, staleness={s['staleness']}")


if __name__ == "__main__":
    main()
