"""Detection limit and precision of I_deep on the synthetic distribution.

The experimental replicates of Comment 3.6 are not yet available, but the
quantities that comment names -- repeatability, relative standard deviation and a
limit of detection -- can be characterised on the synthetic distribution, where
the true Raman component is known exactly and the noise can be held at a fixed
absolute level while the analyte amplitude is varied.

A single pseudo-Voigt band of fixed position and width sits on an empirical
microplate background, with an empirical noise vector rescaled to a fixed standard
deviation. Amplitude is swept from zero. Two conditions are distinguished:

  repeatability   the background is held fixed and only the noise is redrawn
  intermediate    the background is redrawn as well, the analogue of separate wells

Noise is fixed in absolute terms rather than by signal-to-noise ratio; an
SNR-controlled sweep would scale the noise with the analyte and could not define a
detection limit.

    python study_detection_limit.py
"""

import json, os, sys
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import train_ablation as T
from Pmodel import AUSequentialUNet

PEAK_CM, WIDTH_IDX, ETA = 1000.0, 10.0, 0.9
SIGMAS = [0.10, 0.05, 0.025]
AMPS   = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00]
N      = 200
BATCH  = 64
DEV    = T.DEVICE


def i_deep_batch(model, specs):
    """I_deep for a stack of spectra, batched."""
    a = np.asarray(specs, dtype=np.float64)
    lo = a.min(axis=1, keepdims=True); hi = a.max(axis=1, keepdims=True)
    span = (hi - lo).ravel()
    t = torch.tensor((a - lo) / (hi - lo), dtype=torch.float32).unsqueeze(1).to(DEV)
    outs = []
    with torch.no_grad():
        for i in range(0, len(t), BATCH):
            o = model(t[i:i + BATCH])
            outs.append(o[3].squeeze(1).sum(dim=1).cpu().numpy())
    return span / 10.0 * np.concatenate(outs)


def main():
    T.set_seed(0)
    gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                               min_snr=T.TRAIN_MIN_SNR, max_snr=T.TRAIN_MAX_SNR)
    model = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0, kernels=[1, 1, 3],
                             base_latent_dim=16, use_derivatives=False).to(DEV)
    model.load_state_dict(torch.load(os.path.join(BASE, "best_raman2_model.pt"),
                                     map_location=DEV, weights_only=True))
    model.eval()

    x = np.arange(gen.n_points)
    c = gen._find_nearest_idx(PEAK_CM)
    shape = gen.pseudo_voigt(x, c, WIDTH_IDX, ETA)
    nb = np.asarray(gen.noise_bank, dtype=float)
    nb = (nb - nb.mean(axis=1, keepdims=True))
    nb = nb / np.where(nb.std(axis=1, keepdims=True) > 1e-9, nb.std(axis=1, keepdims=True), 1.0)

    fixed_bg = gen.get_mixed_baseline() * 5.0
    bg_pool = np.stack([gen.get_mixed_baseline() * 5.0 for _ in range(N)]) * 1.0

    curves, summary = {}, {}
    for cond in ("repeatability", "intermediate"):
        for sigma in SIGMAS:
            rng = np.random.default_rng(12345)
            rows = []
            for amp in AMPS:
                idx = rng.integers(0, len(nb), N)
                noise = nb[idx] * sigma
                bg = np.repeat(fixed_bg[None, :], N, axis=0) if cond == "repeatability" else bg_pool
                specs = amp * shape[None, :] + bg + noise
                v = i_deep_batch(model, specs)
                rows.append(dict(amp=amp, mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                 rsd=float(100 * v.std(ddof=1) / v.mean()) if abs(v.mean()) > 1e-12 else None))
            curves[f"{cond}_{sigma}"] = rows

            blank = rows[0]
            fit = [r for r in rows if 0.20 <= r["amp"] <= 1.00]
            S, _ = np.polyfit([r["amp"] for r in fit], [r["mean"] for r in fit], 1)
            summary[f"{cond}_{sigma}"] = dict(
                condition=cond, sigma=sigma, blank_sd=blank["sd"], slope=float(S),
                lod_amp=float(3.3 * blank["sd"] / S), loq_amp=float(10.0 * blank["sd"] / S))

    print("=" * 86)
    print(f"I_deep vs analyte amplitude   ({N} realisations per point)")
    print("=" * 86)
    for key, rows in curves.items():
        cond, sigma = key.rsplit("_", 1)
        print(f"\n--- {cond}: noise sd {sigma} (SNR {1/float(sigma):.0f} for a unit peak) ---")
        print(f"{'amp':>6} {'mean I_deep':>13} {'SD':>10} {'RSD %':>9}")
        for r in rows:
            rsd = f"{r['rsd']:9.2f}" if r["rsd"] is not None else "        -"
            print(f"{r['amp']:>6.2f} {r['mean']:>13.3f} {r['sd']:>10.3f} {rsd}")
        s = summary[key]
        print(f"   blank SD {s['blank_sd']:.3f} | slope {s['slope']:.2f} /unit amp "
              f"| LOD {s['lod_amp']:.4f} | LOQ {s['loq_amp']:.4f}")

    with open(os.path.join(BASE, "runs", "detection_limit.json"), "w") as f:
        json.dump(dict(design=dict(peak_cm=PEAK_CM, width_idx=WIDTH_IDX, eta=ETA,
                                   n=N, amps=AMPS, sigmas=SIGMAS),
                       curves=curves, summary=summary), f, indent=2)
    print("\nwrote runs/detection_limit.json")


if __name__ == "__main__":
    main()
