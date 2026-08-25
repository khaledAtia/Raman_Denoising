"""Does the cascade's stage-1 baseline error propagate into its stage-2 signal?

Per-spectrum baseline error and signal error are scored on the frozen validation set
for both models, and the two are correlated. In the cascade the only route from raw
spectrum to final signal passes through the stage-1 output, so a baseline error at
stage 1 is inherited by stage 2. The dual-branch decoders are parallel and share only
the encoder, so no such route exists.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_ablation as T
from Pmodel import AUSequentialUNet
from kazemzadeh_model import CascadedUNet

SIG_SCALE = 10.0
T.set_seed(0)
val_gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                               min_snr=T.VAL_MIN_SNR, max_snr=T.VAL_MAX_SNR,
                               blank_ratio=T.VAL_BLANK_RATIO)
val_ds = T.FixedRamanDataset(val_gen, n_samples=T.VAL_N_SAMPLES)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

DEV = T.DEVICE
ours = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0, kernels=[1, 1, 3], base_latent_dim=16,
                        use_derivatives=False).to(DEV)
ours.load_state_dict(torch.load("best_raman2_model.pt", map_location=DEV, weights_only=True))
ours.eval()
casc = CascadedUNet().to(DEV)
casc.load_state_dict(torch.load(os.path.join("runs", "kazemzadeh_seed0_best.pt"),
                                map_location=DEV, weights_only=True))
casc.eval()

C = {k: [] for k in ("ob", "cb", "os", "cs")}
with torch.no_grad():
    for x, y_s, y_b in val_dl:
        x, y_s, y_b = x.to(DEV), y_s.to(DEV), y_b.to(DEV)
        o = ours(x)
        f, inter, _ = casc(x)
        pm = lambda a, b: (a - b).abs().mean(dim=(1, 2)).cpu().numpy()   # per-spectrum MAE
        C["ob"].append(pm(o[1], y_b))
        C["cb"].append(pm(x - inter / SIG_SCALE, y_b))
        C["os"].append(pm(o[0], y_s))
        C["cs"].append(pm(f, y_s))
A = {k: np.concatenate(v) for k, v in C.items()}

print(f"per-spectrum signal MAE   ours {A['os'].mean():.5f}   cascade {A['cs'].mean():.5f}"
      f"   (Table S12: 0.02564 / 0.04306)")
print()
print("correlation between a spectrum's BASELINE error and its SIGNAL error")
print(f"{'':<14}{'Pearson r':>11}{'Spearman':>11}{'r^2':>9}")
for tag, k in (("Dual-branch", "o"), ("Cascaded", "c")):
    b, s = A[k + "b"], A[k + "s"]
    r = np.corrcoef(b, s)[0, 1]
    rb, rs = b.argsort().argsort(), s.argsort().argsort()
    rho = np.corrcoef(rb, rs)[0, 1]
    print(f"{tag:<14}{r:>11.3f}{rho:>11.3f}{r*r:>9.3f}")

# worst-decile test: how much worse is the signal where the baseline is worst?
print()
for tag, k in (("Dual-branch", "o"), ("Cascaded", "c")):
    b, s = A[k + "b"], A[k + "s"]
    idx = b.argsort()
    lo, hi = idx[:len(idx)//10], idx[-len(idx)//10:]
    print(f"{tag:<14} signal MAE in best-baseline decile {s[lo].mean():.5f}"
          f"   worst-baseline decile {s[hi].mean():.5f}"
          f"   ratio {s[hi].mean()/s[lo].mean():.2f}x")

# ---- paired test: both models see identical inputs, so differencing removes the
# ---- common "hard spectrum" confound that inflates both correlations above.
db = A["cb"] - A["ob"]          # cascade's baseline disadvantage, per spectrum
ds = A["cs"] - A["os"]          # cascade's signal   disadvantage, per spectrum
r = np.corrcoef(db, ds)[0, 1]
rho = np.corrcoef(db.argsort().argsort(), ds.argsort().argsort())[0, 1]
print()
print("PAIRED: cascade's baseline disadvantage vs its signal disadvantage")
print(f"   Pearson r = {r:.3f}   Spearman = {rho:.3f}   r^2 = {r*r:.3f}")

idx = db.argsort()
n = len(idx) // 10
lo, hi = idx[:n], idx[-n:]
print(f"   spectra where cascade's baseline is *least* worse: signal gap {ds[lo].mean():+.5f}")
print(f"   spectra where cascade's baseline is *most*  worse: signal gap {ds[hi].mean():+.5f}")
print(f"   cascade's baseline is better than ours on {(db < 0).mean()*100:.1f}% of spectra;")
print(f"      on those its signal gap is {ds[db < 0].mean():+.5f}, "
      f"vs {ds[db >= 0].mean():+.5f} where its baseline is worse")
