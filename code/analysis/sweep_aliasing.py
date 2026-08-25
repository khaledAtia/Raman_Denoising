"""
R3.5 -- aliasing / sampling diagnostics for the deep-layer quantitative readout.

Reviewer 3:
    "Please discuss aliasing in the deep layer. The Down3 bottleneck spans 512 channels
     and each point covers ~14.8 cm-1, while the peak width is [5, 30]. I wonder whether
     aliasing would occur if the frequency is higher than the Nyquist frequency. It would
     be better to show some data scanning the peak width from 5 to 30 and show how the
     deep-layer integration changes accordingly. Or fix the peak width and shift the peak
     center to check how the integration changes."

Both experiments the reviewer asks for, implemented literally.

WHAT IS MEASURED
    I_deep = (X_max - X_min)/10 * sum_i s_deep_i          (the quantitative readout)

    The deep auxiliary head is supervised against avg_pool1d(target_s, 4), and average
    pooling conserves area, so a perfect head would give

        I_deep  ==  sum(target_s) / 4  =:  A_true

    We therefore report the RATIO I_deep / A_true. A flat ratio across the sweep means the
    readout is unbiased with respect to that variable. Curvature or ripple is the effect
    the reviewer is asking about. The full-resolution output integral is reported as a
    control.

SAMPLING CONTEXT (computed, not assumed)
    input      864 pts -> 1.85 cm-1/pt
    mid head   432 pts -> 3.69 cm-1/pt
    deep head  216 pts -> 7.39 cm-1/pt   <- the readout layer
    bottleneck 108 pts -> 14.8 cm-1/pt   <- the layer the reviewer refers to
    peak FWHM  = width_idx points = width_idx * 1.85 cm-1   (both pseudo-Voigt components)

PAIRED DESIGN
    Within one repetition every sweep point is generated from an identical baseline and
    noise draw, by re-seeding NumPy before each call. Differences along the sweep are then
    attributable to the peak parameter alone and not to background variability.

    python sweep_aliasing.py --arch rk4 --seed 0 --n-rep 20 --snr 25
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import train_ablation as T

SIG_SCALE = 10.0          # generator scales Raman targets by 10 (RamanDataGenerator.py)
POOL_DEEP = 4             # 864 -> 216
CENTER_CM = 1000.0        # isolated peak, mid-range, away from the axis edges
ETA = 0.9
AMPLITUDE = 1.0


@torch.no_grad()
def readout(model, inputs, target_s):
    """Return (I_deep, A_true_deep, I_final, A_true_final) for one raw spectrum."""
    mn, mx = float(np.min(inputs)), float(np.max(inputs))
    span = mx - mn if mx > mn else 1e-6
    x = torch.tensor((inputs - mn) / span, dtype=torch.float32).view(1, 1, -1).to(T.DEVICE)

    pred_s, _pb, _sm, pred_s_deep, _bm, _bd, _xb, _xs = model(x)

    i_deep = float(pred_s_deep.sum().item()) * span / SIG_SCALE
    i_final = float(pred_s.sum().item()) * span / SIG_SCALE

    t = torch.tensor(target_s, dtype=torch.float32).view(1, 1, -1)
    a_deep = float(F.avg_pool1d(t, POOL_DEEP).sum().item())      # == sum(target_s)/4
    a_final = float(t.sum().item())
    return i_deep, a_deep, i_final, a_final


def run_point(model, gen, loc_cm, width_idx, snr, seed):
    """One sweep point under a fixed baseline/noise realisation."""
    np.random.seed(seed)                      # paired design: identical background draw
    inputs, target_s, _tb = gen.generate_defined_spectrum(
        peak_locs_cm=[loc_cm], widths_idx=[width_idx],
        amplitudes=[AMPLITUDE], eta=ETA, snr=snr)
    return readout(model, inputs, target_s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=sorted(T.ARCHITECTURES), default="rk4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rep", type=int, default=20, help="background realisations per point")
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
    print(f"\naxis {axis[0]:.1f}-{axis[-1]:.1f} cm-1 | {len(axis)} pts | {disp:.3f} cm-1/pt")
    print(f"deep head 216 pts -> {disp*4:.2f} cm-1/pt ; bottleneck 108 pts -> {disp*8:.2f} cm-1/pt\n")

    # ==================================================================
    # SWEEP A -- peak width 5..30 at fixed centre
    # ==================================================================
    # Training draws width_idx LOG-UNIFORMLY from [3, 30] points (RamanDataGenerator.py
    # line 1651), so the sweep must start at 3, not 5. Peaks of width_idx 3 have
    # FWHM 5.5 cm-1 = 0.75 samples at the deep head, i.e. genuinely below one sample.
    widths = list(range(3, 31))
    A = {w: [] for w in widths}
    for rep in range(args.n_rep):
        for w in widths:
            i_d, a_d, i_f, a_f = run_point(model, gen, CENTER_CM, w, args.snr, 10_000 + rep)
            A[w].append((i_d / a_d if a_d else np.nan, i_f / a_f if a_f else np.nan))

    print("SWEEP A -- integral ratio vs peak width (readout / area-preserving target)")
    print(f"{'width':>6}{'FWHM cm-1':>11}{'pts/FWHM':>10}{'deep ratio':>16}{'final ratio':>16}")
    rowsA = []
    for w in widths:
        d = np.array([r[0] for r in A[w]]); f = np.array([r[1] for r in A[w]])
        fwhm = w * disp
        print(f"{w:>6}{fwhm:>11.1f}{fwhm/(disp*4):>10.2f}"
              f"{d.mean():>10.3f}+-{d.std():<5.3f}{f.mean():>10.3f}+-{f.std():<5.3f}")
        rowsA.append({"width_idx": w, "fwhm_cm": fwhm, "pts_per_fwhm_deep": fwhm / (disp * 4),
                      "deep_ratio_mean": float(d.mean()), "deep_ratio_std": float(d.std()),
                      "final_ratio_mean": float(f.mean()), "final_ratio_std": float(f.std())})

    # ==================================================================
    # SWEEP B -- sub-sample centre shift, at a narrow and a broad width
    # ==================================================================
    n_steps = 17                       # 16 input points = 4 deep-layer samples
    offsets = np.arange(n_steps) * disp
    rowsB = {}
    print("\nSWEEP B -- integral ratio vs sub-sample peak position")
    for w in (6, 20):
        B = {k: [] for k in range(n_steps)}
        Bf = {k: [] for k in range(n_steps)}
        for rep in range(args.n_rep):
            for k in range(n_steps):
                i_d, a_d, i_f, a_f = run_point(
                    model, gen, CENTER_CM + offsets[k], w, args.snr, 20_000 + rep)
                B[k].append(i_d / a_d if a_d else np.nan)
                Bf[k].append(i_f / a_f if a_f else np.nan)
        means = np.array([np.mean(B[k]) for k in range(n_steps)])
        meansf = np.array([np.mean(Bf[k]) for k in range(n_steps)])
        ripple = (means.max() - means.min()) / means.mean() * 100
        ripplef = (meansf.max() - meansf.min()) / meansf.mean() * 100
        print(f"  width_idx={w:>2} (FWHM {w*disp:.1f} cm-1, {w*disp/(disp*4):.2f} pts/FWHM at deep head)")
        print(f"    deep : mean ratio {means.mean():.3f} | peak-to-peak ripple {ripple:.2f}%")
        print(f"    final: mean ratio {meansf.mean():.3f} | peak-to-peak ripple {ripplef:.2f}%"
              f"   <- control at 864 pts (4x finer)")
        rowsB[str(w)] = {"offsets_cm": offsets.tolist(), "ratio_mean": means.tolist(),
                         "ripple_pct": float(ripple), "mean_ratio": float(means.mean()),
                         "final_ratio_mean": meansf.tolist(),
                         "final_ripple_pct": float(ripplef)}

    # ==================================================================
    # OUTPUT
    # ==================================================================
    os.makedirs(T.RUNS_DIR, exist_ok=True)
    stem = os.path.join(T.RUNS_DIR, f"{tag}_aliasing")
    with open(stem + ".json", "w") as fh:
        json.dump({"checkpoint": ckpt, "snr": args.snr, "n_rep": args.n_rep,
                   "dispersion_cm_per_pt": disp, "deep_cm_per_pt": disp * 4,
                   "bottleneck_cm_per_pt": disp * 8,
                   "sweep_width": rowsA, "sweep_position": rowsB}, fh, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    fw = [r["fwhm_cm"] for r in rowsA]
    dm = np.array([r["deep_ratio_mean"] for r in rowsA])
    ds = np.array([r["deep_ratio_std"] for r in rowsA])
    ax[0].axhline(1.0, color="k", ls=":", lw=1, label="ideal (area preserved)")
    ax[0].fill_between(fw, dm - ds, dm + ds, alpha=0.25, color="crimson")
    ax[0].plot(fw, dm, "o-", color="crimson", ms=3, label="deep-layer readout")
    ax[0].axvline(2 * disp * 4, color="gray", ls="--", lw=1,
                  label="2 samples / FWHM at deep head")
    ax[0].set_xlabel("Peak FWHM (cm$^{-1}$)")
    ax[0].set_ylabel(r"$I_{\rm deep}$ / area-preserving target")
    ax[0].set_title("(a) Width sweep")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    for w, c in (("6", "crimson"), ("20", "dodgerblue")):
        r = rowsB[w]
        ax[1].plot(r["offsets_cm"], r["ratio_mean"], "o-", ms=3, color=c,
                   label=f"width_idx {w} (FWHM {int(w)*disp:.0f} cm$^{{-1}}$), ripple {r['ripple_pct']:.1f}%")
    for k in range(1, 5):
        ax[1].axvline(k * disp * 4, color="gray", ls="--", lw=0.8)
    ax[1].set_xlabel("Peak centre offset (cm$^{-1}$);  dashed = deep-layer sample boundaries")
    ax[1].set_ylabel(r"$I_{\rm deep}$ / area-preserving target")
    ax[1].set_title("(b) Sub-sample position sweep")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(stem + ".png")
    plt.close()
    print(f"\nsaved -> {stem}.json\nsaved -> {stem}.png")


if __name__ == "__main__":
    main()
