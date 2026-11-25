"""Minimal, self-contained CNN in NumPy using im2col + GEMM.

Architecture (54,314 FP32 parameters = 217,256 bytes, matching thinhnd/):
    Input (N, 1, 28, 28)
      -> Conv1 (1 -> 8, 5x5, pad=2) + ReLU + MaxPool(2x2)   -> (N, 8, 14, 14)
      -> Conv2 (8 -> 16, 5x5, pad=2) + ReLU + MaxPool(2x2)  -> (N, 16, 7, 7) -> flatten (N, 784)
      -> Dense1 (784 -> 64) + ReLU
      -> Dense2 (64 -> 10) + Softmax Cross-Entropy

Every phase of forward, backward, and SGD update is timed individually and
accompanied by exact FLOP counts so we can measure where time goes.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np


def im2col(x: np.ndarray, ksize: int = 5, pad: int = 2) -> np.ndarray:
    """Unfold (N, C, H, W) tensor into (C*ksize*ksize, N*H*W) column matrix (stride=1)."""
    n, c, h, w = x.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    cols = np.empty((c, ksize, ksize, n, h, w), dtype=x.dtype)
    for ky in range(ksize):
        for kx in range(ksize):
            cols[:, ky, kx] = x_pad[:, :, ky : ky + h, kx : kx + w].transpose(1, 0, 2, 3)
    return cols.reshape(c * ksize * ksize, n * h * w)


def col2im(dcols: np.ndarray, x_shape: Tuple[int, int, int, int], ksize: int = 5, pad: int = 2) -> np.ndarray:
    """Adjoint of im2col: accumulate column gradients back into (N, C, H, W)."""
    n, c, h, w = x_shape
    dcols_6d = dcols.reshape(c, ksize, ksize, n, h, w)
    dx_pad = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=dcols.dtype)
    for ky in range(ksize):
        for kx in range(ksize):
            dx_pad[:, :, ky : ky + h, kx : kx + w] += dcols_6d[:, ky, kx].transpose(1, 0, 2, 3)
    if pad == 0:
        return dx_pad
    return dx_pad[:, :, pad:-pad, pad:-pad]


def conv_forward_direct_naive(x: np.ndarray, w_4d: np.ndarray, b: np.ndarray, pad: int = 2) -> np.ndarray:
    """Reference nested-loop convolution (CPU baseline to compare against im2col+GEMM)."""
    n, cin, h, w = x.shape
    cout, _, k, _ = w_4d.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out = np.empty((n, cout, h, w), dtype=x.dtype)
    for ni in range(n):
        for co in range(cout):
            for y in range(h):
                for x0 in range(w):
                    acc = float(b[co])
                    for ci in range(cin):
                        for ky in range(k):
                            for kx in range(k):
                                acc += float(w_4d[co, ci, ky, kx]) * float(x_pad[ni, ci, y + ky, x0 + kx])
                    out[ni, co, y, x0] = acc
    return out


def maxpool2x2_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2x2 max pooling with stride 2. Returns (output, argmax_indices)."""
    n, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    patches = x.reshape(n, c, h2, 2, w2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h2, w2, 4)
    idx = np.argmax(patches, axis=-1)
    out = np.take_along_axis(patches, idx[..., None], axis=-1)[..., 0]
    return out, idx


def maxpool2x2_backward(dout: np.ndarray, idx: np.ndarray, x_shape: Tuple[int, int, int, int]) -> np.ndarray:
    """Scatter gradients back to argmax locations."""
    n, c, h, w = x_shape
    h2, w2 = h // 2, w // 2
    dpatches = np.zeros((n, c, h2, w2, 4), dtype=dout.dtype)
    np.put_along_axis(dpatches, idx[..., None], dout[..., None], axis=-1)
    return dpatches.reshape(n, c, h2, w2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


class SmallCNN:
    """5-layer CNN (Conv-Pool-Conv-Pool-FC-FC) with detailed per-op timing & FLOP accounting."""

    def __init__(self, seed: int = 2026):
        rng = np.random.default_rng(seed)
        # He initialization for ReLU layers, Xavier for final logits
        self.params: Dict[str, np.ndarray] = {
            "W1": (rng.standard_normal((8, 25)) * np.sqrt(2.0 / 25.0)).astype(np.float32),
            "b1": np.zeros(8, dtype=np.float32),
            "W2": (rng.standard_normal((16, 200)) * np.sqrt(2.0 / 200.0)).astype(np.float32),
            "b2": np.zeros(16, dtype=np.float32),
            "W3": (rng.standard_normal((64, 784)) * np.sqrt(2.0 / 784.0)).astype(np.float32),
            "b3": np.zeros(64, dtype=np.float32),
            "W4": (rng.standard_normal((10, 64)) * np.sqrt(1.0 / 64.0)).astype(np.float32),
            "b4": np.zeros(10, dtype=np.float32),
        }
        self.velocity: Dict[str, np.ndarray] = {k: np.zeros_like(v) for k, v in self.params.items()}

    @property
    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    @property
    def gradient_bytes(self) -> int:
        return self.num_parameters * 4  # float32

    @staticmethod
    def step_flops(batch_size: int) -> Dict[str, int]:
        """Exact analytical FLOP counts for one forward+backward+SGD step on batch_size images."""
        n = batch_size
        # Conv1: (8, 25) @ (25, N*784); backward dW1 = dZ1 @ cols1.T (no dX needed at input layer)
        conv1_fwd_gemm = 2 * 8 * 25 * (n * 28 * 28)
        conv1_bwd_gemm = 2 * 8 * (n * 28 * 28) * 25

        # Conv2: (16, 200) @ (200, N*196); backward dW2 and dP1
        conv2_fwd_gemm = 2 * 16 * 200 * (n * 14 * 14)
        conv2_bwd_dw_gemm = 2 * 16 * (n * 14 * 14) * 200
        conv2_bwd_dx_gemm = 2 * 200 * 16 * (n * 14 * 14)

        # FC1: (N, 784) @ (784, 64); backward dW3 and dFlat
        fc1_fwd_gemm = 2 * n * 784 * 64
        fc1_bwd_dw_gemm = 2 * 64 * n * 784
        fc1_bwd_dx_gemm = 2 * n * 64 * 784

        # FC2: (N, 64) @ (64, 10); backward dW4 and dH
        fc2_fwd_gemm = 2 * n * 64 * 10
        fc2_bwd_dw_gemm = 2 * 10 * n * 64
        fc2_bwd_dx_gemm = 2 * n * 10 * 64

        conv_gemm = conv1_fwd_gemm + conv1_bwd_gemm + conv2_fwd_gemm + conv2_bwd_dw_gemm + conv2_bwd_dx_gemm
        fc_gemm = fc1_fwd_gemm + fc1_bwd_dw_gemm + fc1_bwd_dx_gemm + fc2_fwd_gemm + fc2_bwd_dw_gemm + fc2_bwd_dx_gemm

        # Non-GEMM arithmetic ops (bias add, ReLU fwd/bwd, maxpool comparisons, softmax+loss, bias sum, SGD)
        bias_relu_pool = (
            (8 * n * 784 * 2) + (8 * n * 196 * 3) + (8 * n * 784)          # conv1 bias+relu, pool1, relu1 bwd
            + (16 * n * 196 * 2) + (16 * n * 49 * 3) + (16 * n * 196)      # conv2 bias+relu, pool2, relu2 bwd
            + (64 * n * 2) + (64 * n)                                      # fc1 bias+relu, relu3 bwd
            + (10 * n)                                                     # fc2 bias
        )
        softmax_xent = 5 * 10 * n
        bias_grads = (8 * n * 784) + (16 * n * 196) + (64 * n) + (10 * n)
        sgd_update = 4 * 54314  # momentum + update

        return {
            "conv_gemm": conv_gemm,
            "fc_gemm": fc_gemm,
            "total_gemm": conv_gemm + fc_gemm,
            "non_gemm_ops": bias_relu_pool + softmax_xent + bias_grads + sgd_update,
            "total_flops": conv_gemm + fc_gemm + bias_relu_pool + softmax_xent + bias_grads + sgd_update,
        }

    def forward_logits(self, x: np.ndarray, profile: bool = False):
        """Run forward pass. If profile=True, also return cache and per-op timing dict (seconds)."""
        timings: Dict[str, float] = {}
        n = x.shape[0]
        p = self.params

        t0 = time.perf_counter()
        cols1 = im2col(x, ksize=5, pad=2)
        t1 = time.perf_counter()
        z1_mat = p["W1"] @ cols1
        t2 = time.perf_counter()
        z1_mat += p["b1"][:, None]
        z1 = z1_mat.reshape(8, n, 28, 28).transpose(1, 0, 2, 3)
        a1 = np.maximum(z1, 0.0)
        p1, idx1 = maxpool2x2_forward(a1)
        t3 = time.perf_counter()

        cols2 = im2col(p1, ksize=5, pad=2)
        t4 = time.perf_counter()
        z2_mat = p["W2"] @ cols2
        t5 = time.perf_counter()
        z2_mat += p["b2"][:, None]
        z2 = z2_mat.reshape(16, n, 14, 14).transpose(1, 0, 2, 3)
        a2 = np.maximum(z2, 0.0)
        p2, idx2 = maxpool2x2_forward(a2)
        flat = p2.reshape(n, 784)
        t6 = time.perf_counter()

        z3 = flat @ p["W3"].T
        t7 = time.perf_counter()
        z3 += p["b3"]
        h = np.maximum(z3, 0.0)
        t8 = time.perf_counter()

        logits = h @ p["W4"].T
        t9 = time.perf_counter()
        logits += p["b4"]
        t10 = time.perf_counter()

        if profile:
            timings["im2col"] = (t1 - t0) + (t4 - t3)
            timings["conv_gemm_fwd"] = (t2 - t1) + (t5 - t4)
            timings["fc_gemm_fwd"] = (t7 - t6) + (t9 - t8)
            timings["relu_pool_bias_fwd"] = (t3 - t2) + (t6 - t5) + (t8 - t7) + (t10 - t9)

        cache = (x, cols1, z1, a1, p1, idx1, cols2, z2, a2, p2, idx2, flat, z3, h)
        return logits, cache, timings

    def step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.05, momentum: float = 0.9,
             apply_update: bool = True, profile: bool = False):
        """One mini-batch forward + backward (+ optional SGD update). Returns (loss, acc, grads, timings)."""
        n = x.shape[0]
        p = self.params
        logits, cache, timings = self.forward_logits(x, profile=profile)
        _, cols1, z1, a1, p1, idx1, cols2, z2, a2, p2, idx2, flat, z3, h = cache

        # Softmax + cross-entropy loss
        t0 = time.perf_counter()
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(shifted)
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)
        loss = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean())
        acc = float((np.argmax(logits, axis=1) == y).mean())
        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        t1 = time.perf_counter()

        # Backward FC2
        dW4 = dlogits.T @ h
        dh = dlogits @ p["W4"]
        t2 = time.perf_counter()
        db4 = dlogits.sum(axis=0)
        dz3 = dh * (z3 > 0.0)
        t3 = time.perf_counter()

        # Backward FC1
        dW3 = dz3.T @ flat
        dflat = dz3 @ p["W3"]
        t4 = time.perf_counter()
        db3 = dz3.sum(axis=0)
        dp2 = dflat.reshape(p2.shape)
        da2 = maxpool2x2_backward(dp2, idx2, a2.shape)
        dz2 = da2 * (z2 > 0.0)
        dz2_mat = dz2.transpose(1, 0, 2, 3).reshape(16, n * 196)
        db2 = dz2_mat.sum(axis=1)
        t5 = time.perf_counter()

        # Backward Conv2
        dW2 = dz2_mat @ cols2.T
        dcols2 = p["W2"].T @ dz2_mat
        t6 = time.perf_counter()
        dp1 = col2im(dcols2, p1.shape, ksize=5, pad=2)
        t7 = time.perf_counter()

        da1 = maxpool2x2_backward(dp1, idx1, a1.shape)
        dz1 = da1 * (z1 > 0.0)
        dz1_mat = dz1.transpose(1, 0, 2, 3).reshape(8, n * 784)
        db1 = dz1_mat.sum(axis=1)
        t8 = time.perf_counter()

        # Backward Conv1 (only dW1 needed)
        dW1 = dz1_mat @ cols1.T
        t9 = time.perf_counter()

        grads = {
            "W1": dW1, "b1": db1,
            "W2": dW2, "b2": db2,
            "W3": dW3, "b3": db3,
            "W4": dW4, "b4": db4,
        }

        t10 = time.perf_counter()
        if apply_update:
            for k in self.params:
                self.velocity[k] = momentum * self.velocity[k] + grads[k]
                self.params[k] -= lr * self.velocity[k]
        t11 = time.perf_counter()

        if profile:
            timings["softmax_loss"] = t1 - t0
            timings["fc_gemm_bwd"] = (t2 - t1) + (t4 - t3)
            timings["conv_gemm_bwd"] = (t6 - t5) + (t9 - t8)
            timings["col2im"] = t7 - t6
            timings["relu_pool_bias_bwd"] = (t3 - t2) + (t5 - t4) + (t8 - t7)
            timings["sgd_update"] = t11 - t10
            timings["total_gemm"] = (
                timings["conv_gemm_fwd"] + timings["conv_gemm_bwd"]
                + timings["fc_gemm_fwd"] + timings["fc_gemm_bwd"]
            )
            timings["total_im2col_col2im"] = timings["im2col"] + timings["col2im"]
            timings["total_elementwise_pool_sgd"] = (
                timings["relu_pool_bias_fwd"] + timings["relu_pool_bias_bwd"]
                + timings["softmax_loss"] + timings["sgd_update"]
            )
            timings["total_step"] = (
                timings["total_gemm"] + timings["total_im2col_col2im"] + timings["total_elementwise_pool_sgd"]
            )

        return loss, acc, grads, timings


def make_synthetic_images(n: int, seed: int = 2026, noise: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Generate (n, 1, 28, 28) float32 synthetic 10-class geometric images + int64 labels."""
    rng = np.random.default_rng(seed)
    templates = np.zeros((10, 28, 28), dtype=np.float32)
    yy, xx = np.mgrid[0:28, 0:28]
    r = np.sqrt((yy - 13.5) ** 2 + (xx - 13.5) ** 2)

    templates[0] = ((r >= 7.0) & (r <= 10.5)).astype(np.float32)                   # ring
    templates[1] = (np.abs(xx - 13.5) <= 2.0).astype(np.float32)                   # vertical bar
    templates[2] = (np.abs(yy - 13.5) <= 2.0).astype(np.float32)                   # horizontal bar
    templates[3] = (np.abs(xx - yy) <= 2.0).astype(np.float32)                     # main diagonal
    templates[4] = (np.abs(xx + yy - 27.0) <= 2.0).astype(np.float32)              # anti-diagonal
    templates[5] = ((np.abs(xx - 13.5) <= 1.8) | (np.abs(yy - 13.5) <= 1.8)).astype(np.float32)  # plus
    templates[6] = ((np.abs(xx - yy) <= 1.8) | (np.abs(xx + yy - 27.0) <= 1.8)).astype(np.float32)  # cross X
    templates[7] = (((yy // 7 + xx // 7) % 2 == 0) & (r <= 12.0)).astype(np.float32)  # checkerboard disk
    templates[8] = ((yy >= 6) & (yy <= 21) & ((np.abs(xx - 7) <= 1.5) | (np.abs(xx - 20) <= 1.5) | (np.abs(yy - 6) <= 1.5) | (np.abs(yy - 21) <= 1.5))).astype(np.float32)  # box
    templates[9] = ((np.abs(yy - 13.5) + np.abs(xx - 13.5) <= 10.0) & (np.abs(yy - 13.5) + np.abs(xx - 13.5) >= 6.5)).astype(np.float32)  # diamond

    labels = rng.integers(0, 10, size=n, dtype=np.int64)
    imgs = templates[labels].copy()
    # Slight random 1px shift per sample so convolution translation invariance helps
    shifts_y = rng.integers(-1, 2, size=n)
    shifts_x = rng.integers(-1, 2, size=n)
    for i in range(n):
        imgs[i] = np.roll(imgs[i], shift=(int(shifts_y[i]), int(shifts_x[i])), axis=(0, 1))
    imgs += rng.normal(0.0, noise, size=imgs.shape).astype(np.float32)
    return imgs[:, None, :, :], labels


def check_gradients(seed: int = 7) -> Dict[str, float]:
    """Finite-difference verification of backward pass (uses float64 copy for precision)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((4, 1, 28, 28)).astype(np.float64)
    y = np.array([0, 3, 5, 8], dtype=np.int64)

    net = SmallCNN(seed=seed)
    for k in net.params:
        net.params[k] = net.params[k].astype(np.float64)

    _, _, grads, _ = net.step(x, y, apply_update=False, profile=False)
    eps = 1e-6
    max_rel_err = 0.0
    per_param: Dict[str, float] = {}

    for name in ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"):
        param = net.params[name]
        flat_p = param.ravel()
        flat_g = grads[name].ravel()
        sample_idx = np.linspace(0, flat_p.size - 1, num=min(8, flat_p.size), dtype=int)
        worst = 0.0
        for idx in sample_idx:
            orig = flat_p[idx]
            flat_p[idx] = orig + eps
            lp, _, _, _ = net.step(x, y, apply_update=False)
            flat_p[idx] = orig - eps
            lm, _, _, _ = net.step(x, y, apply_update=False)
            flat_p[idx] = orig
            num_g = (lp - lm) / (2.0 * eps)
            ana_g = flat_g[idx]
            denom = max(1e-8, abs(num_g) + abs(ana_g))
            rel = abs(num_g - ana_g) / denom
            worst = max(worst, float(rel))
        per_param[name] = worst
        max_rel_err = max(max_rel_err, worst)

    return {"max_rel_err": max_rel_err, "per_param": per_param}


if __name__ == "__main__":
    res = check_gradients()
    print(f"Gradient finite-diff max rel error: {res['max_rel_err']:.3e}")
    for k, v in res["per_param"].items():
        print(f"  {k}: {v:.3e}")
