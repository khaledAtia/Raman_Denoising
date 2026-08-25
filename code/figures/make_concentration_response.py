"""SI figure: does the recovered signal scale with concentration?

Three methods on the glycerol series of Figure 8 (Spectrum/Glycerin2):
the dual-branch network, the cascaded U-Net trained on the same data, and the
conventional airPLS + Savitzky-Golay pipeline. A proportional response is a
prediction, not a fit, so the test is the deviation from 1.00 / 1.50 / 2.00 and
no regression coefficient is quoted.

    python make_concentration_response.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter

BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(BASE, "paper_revised", "figures")
sys.path.insert(0, BASE)
from Pmodel import AUSequentialUNet
from kazemzadeh_model import CascadedUNet
from RamanUtils.airPLS import airPLSz

CONCS = np.array([100.0, 150.0, 200.0])
LO_X, HI_X = 320.0, 1700.0
SG_WIN, SG_ORD = 9, 3
BANDS = [495, 852, 1054, 1466]
TMPL = os.path.join(BASE, "Spectrum", "Glycerin2", "{c}_P.txt")

C_OURS, C_CASC, C_CONV = "#08519c", "#d95f02", "#6b6b6b"

ours = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0, kernels=[1, 1, 3],
                        base_latent_dim=16, use_derivatives=False)
ours.load_state_dict(torch.load(os.path.join(BASE, "best_raman2_model.pt"),
                                map_location="cpu", weights_only=True))
ours.eval()
casc = CascadedUNet()
casc.load_state_dict(torch.load(os.path.join(BASE, "runs", "kazemzadeh_seed0_best.pt"),
                                map_location="cpu", weights_only=True))
casc.eval()

integ = {k: [] for k in ("ours", "cascade", "conv", "conv_raw", "ours_full")}
bandamp = {k: {b: [] for b in BANDS} for k in ("ours", "cascade", "conv")}

for c in CONCS:
    d = np.loadtxt(TMPL.format(c=int(c)), skiprows=1)
    x, raw = d[:, 0], d[:, 1]
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    t = torch.tensor((raw - lo) / span, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        o = ours(t)
        f, _i, _b = casc(t)

    m = (x >= LO_X) & (x <= HI_X)
    o_s = o[0].squeeze().numpy() * span / 10.0
    c_s = f.squeeze().numpy() * span / 10.0
    a_raw = airPLSz(raw, 1e4, 2, 40)
    a_sg = savgol_filter(a_raw, SG_WIN, SG_ORD)

    integ["ours"].append(span / 10.0 * o[3].squeeze().numpy().sum())   # I_deep
    integ["ours_full"].append(o_s[m].sum())
    integ["cascade"].append(c_s[m].sum())
    integ["conv"].append(a_sg[m].sum())
    integ["conv_raw"].append(a_raw[m].sum())

    for b in BANDS:
        w = np.abs(x - b) <= 8
        bandamp["ours"][b].append(float(o_s[w].max()))
        bandamp["cascade"][b].append(float(c_s[w].max()))
        bandamp["conv"][b].append(float(a_sg[w].max()))

norm = {k: np.array(v) / v[0] for k, v in integ.items()}

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=150,
                       gridspec_kw={"width_ratios": [1.0, 1.15]})

ideal = CONCS / CONCS[0]
ax[0].plot(CONCS, ideal, color="0.55", lw=1.2, ls="--", zorder=1,
           label="proportional response")
for key, col, lab in (("ours", C_OURS, "dual-branch, $I_{\\mathrm{deep}}$"),
                      ("cascade", C_CASC, "cascaded U-Net"),
                      ("conv", C_CONV, "airPLS + Savitzky--Golay")):
    ax[0].plot(CONCS, norm[key], color=col, lw=2.0, marker="o", ms=7, zorder=3, label=lab)
ax[0].set_xticks(CONCS)
ax[0].set_xlabel("Glycerol concentration (mM)")
ax[0].set_ylabel("Integrated recovered signal\n(normalised to 100 mM)")
ax[0].set_title("(a)  Response vs concentration", fontsize=10, loc="left")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
for s in ("top", "right"):
    ax[0].spines[s].set_visible(False)
ax[0].grid(axis="y", color="0.92", lw=0.7)
ax[0].set_axisbelow(True)

idx = np.arange(len(BANDS))
w = 0.26
for k, (key, col, lab) in enumerate((("ours", C_OURS, "dual-branch"),
                                     ("cascade", C_CASC, "cascade"),
                                     ("conv", C_CONV, "airPLS + SG"))):
    r = [bandamp[key][b][2] / max(bandamp[key][b][0], 1e-9) for b in BANDS]
    ax[1].bar(idx + (k - 1) * w, r, w, color=col, label=lab, zorder=3)
ax[1].axhline(2.0, color="0.55", lw=1.2, ls="--", zorder=2)
ax[1].annotate("proportional (2.0)", xy=(len(BANDS) - 0.5, 2.0),
               xytext=(0, 5), textcoords="offset points",
               ha="right", fontsize=8.5, color="0.4")
ax[1].set_xticks(idx)
ax[1].set_xticklabels([f"{b}" for b in BANDS])
ax[1].set_xlabel("Glycerol band (cm$^{-1}$)")
ax[1].set_ylabel("Amplitude ratio, 200 mM / 100 mM")
ax[1].set_title("(b)  Band-level response, 200 mM relative to 100 mM",
                fontsize=10, loc="left")
ax[1].legend(frameon=False, fontsize=8.5)
for s in ("top", "right"):
    ax[1].spines[s].set_visible(False)
ax[1].grid(axis="y", color="0.92", lw=0.7)
ax[1].set_axisbelow(True)

fig.tight_layout()
out = os.path.join(FIGS, "concentration_response.png")
fig.savefig(out)
print(f"wrote {out}\n")

print("Integrated response, normalised to 100 mM (ideal 1.00 / 1.50 / 2.00)")
for k in ("ours", "ours_full", "cascade", "conv", "conv_raw"):
    print(f"  {k:>10}: " + "  ".join(f"{v:5.2f}" for v in norm[k]))
print()
print("Band-level 200/100 amplitude ratio (ideal 2.00)")
print(f"  {'band':>6} {'ours':>7} {'casc':>7} {'conv':>7}")
for b in BANDS:
    print(f"  {b:>6}" + "".join(
        f" {bandamp[k][b][2] / max(bandamp[k][b][0], 1e-9):>7.2f}"
        for k in ("ours", "cascade", "conv")))
