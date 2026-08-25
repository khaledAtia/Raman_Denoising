"""Run the dual-branch network and the cascaded U-Net on the experimental
glycerol spectra, so the two learned methods can be compared on real data rather
than only on the synthetic validation set.

Both models were trained on the identical synthetic engine from the same seed.
Preprocessing here reproduces inference exactly as in training: min-max
normalisation of the raw spectrum, with the signal target carrying a factor of
ten, so predictions are returned to counts by (X_max - X_min) / 10.

The dual-branch prediction stored in Spectrum/Glycerin/*_processed.txt is used to
verify the pipeline before the cascade is run through it.

    python compare_glycerol_models.py
"""

import os
import sys

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from Pmodel import AUSequentialUNet
from kazemzadeh_model import CascadedUNet

DEV = torch.device("cpu")          # 3 spectra; keep the GPU free for training
CONCS = (100, 150, 200)
OURS_CKPT = os.path.join(BASE, "best_raman2_model.pt")
CASC_CKPT = os.path.join(BASE, "runs", "kazemzadeh_seed0_best.pt")


def load_spectrum(c):
    d = np.loadtxt(os.path.join(BASE, "Spectrum", "Glycerin", f"{c}_processed.txt"),
                   skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]   # shift, raw, stored_raman, stored_base


def normalise(raw):
    lo, hi = float(np.min(raw)), float(np.max(raw))
    return (raw - lo) / (hi - lo), lo, hi


def to_counts(pred_norm, lo, hi):
    """Undo the min-max normalisation and the x10 target scaling."""
    return pred_norm * (hi - lo) / 10.0


# ------------------------------------------------------------------ models
ours = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0,
                        kernels=[1, 1, 3], base_latent_dim=16,
                        use_derivatives=False).to(DEV)
ours.load_state_dict(torch.load(OURS_CKPT, map_location=DEV, weights_only=True))
ours.eval()

casc = CascadedUNet().to(DEV)
casc.load_state_dict(torch.load(CASC_CKPT, map_location=DEV, weights_only=True))
casc.eval()

print(f"dual-branch : {sum(p.numel() for p in ours.parameters()):,} parameters")
print(f"cascade     : {sum(p.numel() for p in casc.parameters()):,} parameters\n")

out = {}
print("=" * 74)
print("PIPELINE CHECK -- recomputed vs stored dual-branch prediction")
print("=" * 74)
for c in CONCS:
    x, raw, stored_raman, stored_base = load_spectrum(c)
    norm, lo, hi = normalise(raw)
    t = torch.tensor(norm, dtype=torch.float32).view(1, 1, -1).to(DEV)

    with torch.no_grad():
        o = ours(t)
        ours_s = to_counts(o[0].squeeze().cpu().numpy(), lo, hi)
        ours_b = o[1].squeeze().cpu().numpy() * (hi - lo) + lo

        final, inter, casc_b = casc(t)
        casc_s = to_counts(final.squeeze().cpu().numpy(), lo, hi)
        casc_i = to_counts(inter.squeeze().cpu().numpy(), lo, hi)
        casc_bb = casc_b.squeeze().cpu().numpy() * (hi - lo) + lo

    resid = np.abs(ours_s - stored_raman)
    denom = max(stored_raman.max(), 1e-9)
    print(f"  {c} mM: max |recomputed - stored| = {resid.max():8.2f} counts "
          f"({100 * resid.max() / denom:5.2f}% of peak)   "
          f"corr = {np.corrcoef(ours_s, stored_raman)[0, 1]:.6f}")

    out[c] = dict(x=x, raw=raw, ours_s=ours_s, ours_b=ours_b,
                  casc_s=casc_s, casc_i=casc_i, casc_b=casc_bb,
                  stored=stored_raman)

np.savez(os.path.join(BASE, "runs", "glycerol_model_comparison.npz"),
         **{f"{c}_{k}": v for c in CONCS for k, v in out[c].items()})
print(f"\nwrote runs/glycerol_model_comparison.npz")

print()
print("=" * 74)
print("RECOVERED AMPLITUDE AT THE REPORTED GLYCEROL BANDS (counts)")
print("=" * 74)
BANDS = [495, 673, 852, 1054, 1115, 1162, 1466]
print(f"{'band':>6}" + "".join(f"{c:>10} mM" for c in CONCS))
print(f"{'':>6}" + "  ours  casc" * 3)
for b in BANDS:
    row = f"{b:>6}"
    for c in CONCS:
        d = out[c]
        w = np.abs(d["x"] - b) <= 8
        row += f" {d['ours_s'][w].max():6.0f}{d['casc_s'][w].max():6.0f}"
    print(row)
