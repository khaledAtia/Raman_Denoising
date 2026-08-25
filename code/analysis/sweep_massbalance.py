"""
Where does the missing area of a broad band go?

Section S7 reports that broad Raman bands are under-reported by up to 18%, and attributes
this to the baseline branch absorbing them. That attribution was an interpretation, not a
measurement. This script measures it directly, by mass balance, with no retraining.

METHOD -- exact paired differencing
    The generator returns a spectrum as X = S + B + N. For one realisation we form

        X_with    = X            (peak present)
        X_without = X - S        (identical baseline AND identical noise, peak removed)

    so the pair differs in the peak alone -- not merely in a re-drawn background. Running
    the model on both and differencing its two outputs isolates the network's response to
    the peak:

        dSignal   = area of predicted Raman   (with) - (without)
        dBaseline = area of predicted baseline (with) - (without)

    If the peak is fully accounted for,  dSignal + dBaseline = A_true = area of S.
    The split between the two terms is the quantity of interest: dBaseline/A_true is the
    fraction of the band absorbed by the baseline branch.

WHY THIS IS THE RIGHT TEST
    Integrating over a window around the peak would require choosing a window, and a
    Lorentzian tail has no natural cut-off. Paired differencing removes the background
    exactly and needs no window at all.

    python sweep_massbalance.py --arch rk4 --seed 0 --n-rep 25 --snr 25
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import train_ablation as T

SIG_SCALE = 10.0
CENTER_CM = 1000.0
ETA = 0.9
AMPLITUDE = 1.0


@torch.no_grad()
def predict(model, raw):
    """Run the model on a raw spectrum; return (signal_area, baseline_area) in counts."""
    mn, mx = float(np.min(raw)), float(np.max(raw))
    span = mx - mn if mx > mn else 1e-6
    x = torch.tensor((raw - mn) / span, dtype=torch.float32).view(1, 1, -1).to(T.DEVICE)
    pred_s, pred_b, _sm, pred_sd, _bm, _bd, _xb, _xs = model(x)

    sig_area = float(pred_s.sum().item()) * span / SIG_SCALE
    deep_area = float(pred_sd.sum().item()) * span / SIG_SCALE
    # baseline head predicts the normalised baseline: b_raw = b_norm*span + min
    base_area = float(pred_b.sum().item()) * span + mn * pred_b.shape[-1]
    return sig_area, base_area, deep_area


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=sorted(T.ARCHITECTURES), default="rk4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rep", type=int, default=25)
    p.add_argument("--snr", type=float, default=25.0)
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    tag = f"{args.arch}_seed{args.seed}"
    ckpt = args.checkpoint or os.path.join(T.RUNS_DIR, f"{tag}_best.pt")

    model = T.build_model(args.arch, "subtract").to(T.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    model.eval()

    gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                               min_snr=T.VAL_MIN_SNR, max_snr=T.VAL_MAX_SNR)
    axis = gen.raman_shift if hasattr(gen, "raman_shift") else gen._raman_shift
    disp = float((axis[-1] - axis[0]) / (len(axis) - 1))

    widths = list(range(3, 31))
    acc = {w: [] for w in widths}

    for rep in range(args.n_rep):
        for w in widths:
            np.random.seed(30_000 + rep)          # identical background per rep
            raw, target_s, _tb = gen.generate_defined_spectrum(
                peak_locs_cm=[CENTER_CM], widths_idx=[w],
                amplitudes=[AMPLITUDE], eta=ETA, snr=args.snr)

            a_true = float(target_s.sum())
            if a_true <= 0:
                continue

            s_with, b_with, d_with = predict(model, raw)
            s_wo, b_wo, d_wo = predict(model, raw - target_s)  # peak removed, same B and N

            acc[w].append((
                (s_with - s_wo) / a_true,      # fraction recovered by the Raman head
                (b_with - b_wo) / a_true,      # fraction absorbed by the baseline head
                (d_with - d_wo) / a_true * 4,  # deep readout, x4 for the pooling factor
            ))

    print(f"\ncheckpoint : {ckpt}")
    print(f"{args.n_rep} paired realisations per width, SNR {args.snr}\n")
    print("Fraction of the true peak area accounted for by each branch")
    print(f"{'width':>6}{'FWHM':>8}{'recovered':>20}{'absorbed by base':>20}{'unaccounted':>14}")

    rows = []
    for w in widths:
        a = np.array(acc[w])
        if not len(a):
            continue
        sig, base = a[:, 0], a[:, 1]
        unacc = 1.0 - sig.mean() - base.mean()
        print(f"{w:>6}{w*disp:>8.1f}{sig.mean():>13.3f}+-{sig.std():<6.3f}"
              f"{base.mean():>13.3f}+-{base.std():<6.3f}{unacc:>14.3f}")
        rows.append({"width_idx": w, "fwhm_cm": w * disp,
                     "recovered_mean": float(sig.mean()), "recovered_std": float(sig.std()),
                     "absorbed_mean": float(base.mean()), "absorbed_std": float(base.std()),
                     "unaccounted": float(unacc),
                     "deep_mean": float(a[:, 2].mean())})

    os.makedirs(T.RUNS_DIR, exist_ok=True)
    stem = os.path.join(T.RUNS_DIR, f"{tag}_massbalance")
    with open(stem + ".json", "w") as fh:
        json.dump({"checkpoint": ckpt, "snr": args.snr, "n_rep": args.n_rep,
                   "dispersion_cm_per_pt": disp, "rows": rows}, fh, indent=2)

    fw = [r["fwhm_cm"] for r in rows]
    rec = np.array([r["recovered_mean"] for r in rows])
    ab = np.array([r["absorbed_mean"] for r in rows])
    plt.figure(figsize=(6.5, 4.2), dpi=150)
    plt.axhline(1.0, color="k", ls=":", lw=1)
    plt.plot(fw, rec, "o-", ms=3, color="crimson", label="recovered by Raman head")
    plt.plot(fw, ab, "s-", ms=3, color="dodgerblue", label="absorbed by baseline head")
    plt.plot(fw, rec + ab, "^--", ms=3, color="gray", label="sum (mass balance)")
    plt.xlabel("Peak FWHM (cm$^{-1}$)")
    plt.ylabel("Fraction of true peak area")
    plt.title("Where the area of a band goes")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(stem + ".png"); plt.close()
    print(f"\nsaved -> {stem}.json\nsaved -> {stem}.png")


if __name__ == "__main__":
    main()
