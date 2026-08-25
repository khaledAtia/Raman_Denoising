"""
SNR-robustness comparison: dual-branch network against the cascaded U-Net.

Reproduces the protocol used for Figure 6 of the manuscript and applies it to both models,
so that the resulting curves are directly comparable:

    SNR levels        5.0 to 25.0 in steps of 2.5
    spectra per level 500, drawn from the same generator with the SNR locked to one value
    metrics           per-sample MSE, MAE and cosine similarity of the recovered Raman
                      signal against the ground truth, reported as mean and standard
                      deviation over the 500 spectra

Both models are evaluated on a BIT-IDENTICAL set of spectra at every SNR: the generator is
re-seeded immediately before each level, so the two models see the same 500 spectra rather
than two independent draws. Differences between the curves are therefore attributable to
the models alone.

The cascaded network produces two outputs, an intermediate baseline-corrected spectrum and
a final denoised spectrum. It is scored on its FINAL output, which is the counterpart of
our single Raman output and the one its authors treat as the result of the pipeline.

    python compare_snr_robustness.py
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
from torch.utils.data import DataLoader

import train_ablation as T
from kazemzadeh_model import CascadedUNet

SIG_SCALE = 10.0
N_PER_LEVEL = 500


def load_ours(ckpt):
    m = T.build_model("rk4", "subtract").to(T.DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    return m.eval()


def load_cascade(ckpt):
    m = CascadedUNet().to(T.DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    return m.eval()


@torch.no_grad()
def evaluate(models, snr_levels, seed, batch_size=64):
    """Return {name: {metric: (means, stds)}} over the SNR sweep."""
    acc = {name: {"mse": [], "mae": [], "cos": [], "mse_sd": [], "mae_sd": [], "cos_sd": []}
           for name in models}

    for snr in snr_levels:
        # re-seed so every model sees the SAME 500 spectra at this SNR
        T.set_seed(seed + int(snr * 10))
        gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                                   epoch_size=N_PER_LEVEL,
                                   min_snr=float(snr), max_snr=float(snr),
                                   blank_ratio=0.0)
        batches = [(x, y) for x, y, _b in DataLoader(gen, batch_size=batch_size, shuffle=False)]

        for name, model in models.items():
            mse, mae, cos = [], [], []
            for x, y in batches:
                x, y = x.to(T.DEVICE), y.to(T.DEVICE)
                out = model(x)
                pred = out[0]                       # ours: final signal; cascade: final signal
                n = x.size(0)
                mse.extend(F.mse_loss(pred, y, reduction="none").view(n, -1).mean(1).cpu().numpy())
                mae.extend(F.l1_loss(pred, y, reduction="none").view(n, -1).mean(1).cpu().numpy())
                cos.extend(F.cosine_similarity(pred.flatten(1), y.flatten(1), dim=1).cpu().numpy())
            for k, v in (("mse", mse), ("mae", mae), ("cos", cos)):
                acc[name][k].append(float(np.mean(v)))
                acc[name][k + "_sd"].append(float(np.std(v)))
        print(f"  SNR {snr:4.1f}  " + "  ".join(
            f"{n}: MSE {acc[n]['mse'][-1]:.5f} cos {acc[n]['cos'][-1]:.4f}" for n in models))
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ours", default=os.path.join(T.RUNS_DIR, "rk4_seed0_best.pt"))
    p.add_argument("--cascade", default=os.path.join(T.RUNS_DIR, "kazemzadeh_seed0_best.pt"))
    args = p.parse_args()

    snr_levels = np.arange(5.0, 27.5, 2.5)
    models = {"ours": load_ours(args.ours), "cascade": load_cascade(args.cascade)}
    print(f"Evaluating {len(snr_levels)} SNR levels x {N_PER_LEVEL} spectra, identical draws\n")
    acc = evaluate(models, snr_levels, args.seed)

    stem = os.path.join(T.RUNS_DIR, "snr_comparison")
    with open(stem + ".json", "w") as f:
        json.dump({"snr_levels": snr_levels.tolist(), "n_per_level": N_PER_LEVEL,
                   "checkpoints": {"ours": args.ours, "cascade": args.cascade},
                   "results": acc}, f, indent=2)

    # --- figure: three stacked panels, matching the layout of Figure 6 ---------
    fig, ax = plt.subplots(3, 1, figsize=(6.2, 8.4), dpi=150, sharex=True)
    style = {"ours": dict(color="crimson", marker="o", label="Dual-branch (this work)"),
             "cascade": dict(color="dodgerblue", marker="s", label="Cascaded U-Net")}

    for i, (key, lab) in enumerate([("mse", "MSE"), ("mae", "MAE"),
                                    ("cos", "Cosine similarity")]):
        for name in models:
            m = np.array(acc[name][key]); sd = np.array(acc[name][key + "_sd"])
            st = style[name]
            ax[i].plot(snr_levels, m, ms=4, lw=1.6, **st)
            ax[i].fill_between(snr_levels, m - sd, m + sd, color=st["color"], alpha=0.15)
        ax[i].set_ylabel(lab)
        ax[i].grid(alpha=0.3)
        if key != "cos":
            ax[i].set_yscale("log")
    ax[0].legend(fontsize=8, loc="upper right")
    ax[2].set_xlabel("Signal-to-noise ratio")
    plt.tight_layout()
    plt.savefig(stem + ".png")
    plt.close()

    print(f"\nsaved -> {stem}.json\nsaved -> {stem}.png")

    print("\nSummary (mean over 500 spectra per level):")
    print(f"{'SNR':>6}{'MSE ours':>12}{'MSE casc':>12}{'cos ours':>11}{'cos casc':>11}")
    for i, snr in enumerate(snr_levels):
        print(f"{snr:>6.1f}{acc['ours']['mse'][i]:>12.5f}{acc['cascade']['mse'][i]:>12.5f}"
              f"{acc['ours']['cos'][i]:>11.4f}{acc['cascade']['cos'][i]:>11.4f}")


if __name__ == "__main__":
    main()
