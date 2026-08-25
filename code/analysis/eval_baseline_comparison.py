"""Baseline-branch accuracy: dual-branch vs. cascaded U-Net (Kazemzadeh et al.).

Both models are scored on the identical frozen validation set used for Table S12.

  ours   : forward() returns (signal, baseline, ...) -- baseline is predicted directly.
  cascade: forward() returns (final, intermediate, _). The intermediate is supervised
           on (x - y_b) * SIG_SCALE, so the implied baseline is x - intermediate/SIG_SCALE.
           NOTE the module's own `predicted_baseline = x - intermediate` omits the /10 and
           is dimensionally wrong; it is unused elsewhere and is not used here.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_ablation as T
from Pmodel import AUSequentialUNet
from kazemzadeh_model import CascadedUNet

SIG_SCALE = 10.0
SEED = 0

T.set_seed(SEED)
val_gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                               min_snr=T.VAL_MIN_SNR, max_snr=T.VAL_MAX_SNR,
                               blank_ratio=T.VAL_BLANK_RATIO)
val_ds = T.FixedRamanDataset(val_gen, n_samples=T.VAL_N_SAMPLES)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
print(f"frozen validation set: {len(val_ds)} spectra "
      f"(SNR {T.VAL_MIN_SNR}-{T.VAL_MAX_SNR}, blank ratio {T.VAL_BLANK_RATIO})")

DEV = T.DEVICE
ours = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0, kernels=[1, 1, 3],
                        base_latent_dim=16, use_derivatives=False).to(DEV)
ours.load_state_dict(torch.load("best_raman2_model.pt", map_location=DEV, weights_only=True))
ours.eval()

casc = CascadedUNet().to(DEV)
casc.load_state_dict(torch.load(os.path.join("runs", "kazemzadeh_seed0_best.pt"),
                                map_location=DEV, weights_only=True))
casc.eval()

acc = {k: [] for k in ("ours_mae", "casc_mae", "ours_rmse", "casc_rmse",
                       "ours_max", "casc_max", "span")}
with torch.no_grad():
    for x, y_s, y_b in val_dl:
        x, y_b = x.to(DEV), y_b.to(DEV)
        b_ours = ours(x)[1]
        inter = casc(x)[1]
        b_casc = x - inter / SIG_SCALE

        for tag, b in (("ours", b_ours), ("casc", b_casc)):
            e = (b - y_b).abs()
            acc[f"{tag}_mae"].append(e.mean(dim=(1, 2)).cpu().numpy())
            acc[f"{tag}_rmse"].append(((b - y_b) ** 2).mean(dim=(1, 2)).sqrt().cpu().numpy())
            acc[f"{tag}_max"].append(e.amax(dim=(1, 2)).cpu().numpy())
        acc["span"].append((y_b.amax(dim=(1, 2)) - y_b.amin(dim=(1, 2))).cpu().numpy())

A = {k: np.concatenate(v) for k, v in acc.items()}

print()
print(f"{'metric':<34}{'Dual-branch':>13}{'Cascaded':>13}{'Difference':>13}")
print("-" * 73)
for name, key in (("Baseline MAE (normalised)", "mae"),
                  ("Baseline RMSE (normalised)", "rmse"),
                  ("Baseline max abs. error", "max")):
    o, c = A[f"ours_{key}"].mean(), A[f"casc_{key}"].mean()
    print(f"{name:<34}{o:>13.5f}{c:>13.5f}{(c - o) / o * 100:>12.1f}%")

o_rel = (A["ours_mae"] / A["span"]).mean() * 100
c_rel = (A["casc_mae"] / A["span"]).mean() * 100
print(f"{'Baseline MAE / baseline span':<34}{o_rel:>12.2f}%{c_rel:>12.2f}%{(c_rel-o_rel)/o_rel*100:>12.1f}%")

win = (A["ours_mae"] < A["casc_mae"]).mean() * 100
print()
print(f"dual-branch baseline is closer on {win:.1f}% of the {len(A['ours_mae'])} validation spectra")
