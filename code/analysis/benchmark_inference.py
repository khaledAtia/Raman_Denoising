"""
Inference-cost benchmark for the throughput claims.

Measures, for each method, the time to process a single spectrum and a full 96-well plate,
on GPU and on CPU, together with parameter counts and the number of forward passes each
method requires. The classical airPLS + Savitzky-Golay pipeline is included as the
reference against which the deep methods are compared.

Latencies are reported as the mean over repeated runs after a warm-up, with the GPU
synchronised around the timed region so that asynchronous kernel launches are not mistaken
for fast execution.

    python benchmark_inference.py
"""

import json
import os
import time

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

import train_ablation as T
from kazemzadeh_model import CascadedUNet

N_POINTS = 864
WARMUP, ITERS = 10, 50


# ---------------------------------------------------------------- airPLS
def airPLS(y, lam=100, order=1, wep=0.1, p=0.05, itermax=15):
    """Adaptive iteratively reweighted penalised least squares (Zhang et al. 2010)."""
    m = y.shape[0]
    w = np.ones(m)
    D = eye(m, format="csc")
    for _ in range(order):
        D = D[1:] - D[:-1]
    DTD = lam * (D.T @ D)
    z = y.copy()
    for i in range(1, itermax + 1):
        W = diags(w, 0, shape=(m, m), format="csc")
        z = spsolve(csc_matrix(W + DTD), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0:
            break
        s = np.abs(dn).sum()
        if s < wep * np.abs(y).sum():
            break
        w = np.zeros(m)
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / s)
    return z


def time_classical(batch):
    x = np.random.rand(batch, N_POINTS) + 1.0
    for _ in range(3):
        for row in x:
            savgol_filter(row - airPLS(row), 11, 3)
    t0 = time.perf_counter()
    reps = max(1, ITERS // 10)
    for _ in range(reps):
        for row in x:
            savgol_filter(row - airPLS(row), 11, 3)
    return (time.perf_counter() - t0) / reps * 1e3


@torch.no_grad()
def time_model(model, batch, device):
    model.eval().to(device)
    x = torch.randn(batch, 1, N_POINTS, device=device)
    for _ in range(WARMUP):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3


def main():
    gpu = T.DEVICE
    cpu = torch.device("cpu")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    variants = {
        "dual-branch (this work)": (T.build_model("rk4", "subtract"), 1),
        "dual-branch, plain encoder": (T.build_model("plain", "subtract"), 1),
        "cascaded U-Net": (CascadedUNet(), 2),
    }

    rows = []
    for name, (m, passes) in variants.items():
        n = sum(p.numel() for p in m.parameters())
        r = {"method": name, "params": n, "passes": passes}
        for dev, tag in ((gpu, "gpu"), (cpu, "cpu")):
            if dev.type == "cuda" and not torch.cuda.is_available():
                continue
            r[f"{tag}_b1_ms"] = round(time_model(m, 1, dev), 3)
            r[f"{tag}_plate_ms"] = round(time_model(m, 96, dev), 3)
        rows.append(r)
        print(f"  {name:<28} params {n:>10,}  "
              f"GPU {r.get('gpu_b1_ms', float('nan')):.2f}/{r.get('gpu_plate_ms', float('nan')):.1f} ms  "
              f"CPU {r.get('cpu_b1_ms', float('nan')):.1f}/{r.get('cpu_plate_ms', float('nan')):.0f} ms")

    print("  timing airPLS + Savitzky-Golay (CPU only) ...")
    c1 = time_classical(1)
    c96 = time_classical(96)
    rows.append({"method": "airPLS + Savitzky-Golay", "params": None, "passes": None,
                 "cpu_b1_ms": round(c1, 3), "cpu_plate_ms": round(c96, 3)})
    print(f"  {'airPLS + Savitzky-Golay':<28} {'--':>17}  "
          f"{'':>22}CPU {c1:.1f}/{c96:.0f} ms")

    out = os.path.join(T.RUNS_DIR, "inference_benchmark.json")
    with open(out, "w") as f:
        json.dump({"n_points": N_POINTS, "warmup": WARMUP, "iters": ITERS,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "rows": rows}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
