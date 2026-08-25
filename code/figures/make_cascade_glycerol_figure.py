"""SI figure: dual-branch network vs the cascaded U-Net on experimental glycerol.

Both networks were trained on the identical synthetic engine from the same seed
and are evaluated here through one identical inference path, so the comparison is
internally consistent. airPLS, which is independent of both, provides a neutral
reference. The manuscript's Figure 8 is not touched.

    python make_cascade_glycerol_figure.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(BASE, "paper_revised", "figures")
sys.path.insert(0, BASE)

from Pmodel import AUSequentialUNet
from kazemzadeh_model import CascadedUNet
from RamanUtils.airPLS import airPLSz

DEV = torch.device("cpu")
CONCS = (100, 150, 200)
BANDS = [495, 673, 819, 852, 925, 976, 1054, 1115, 1162, 1270, 1466]

C_REF, C_OURS, C_CASC = "0.62", "#08519c", "#d95f02"


def load(c):
    d = np.loadtxt(os.path.join(BASE, "Spectrum", "Glycerin2", f"{c}_P.txt"),
                   skiprows=1)
    return d[:, 0], d[:, 1]


ours = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0, kernels=[1, 1, 3],
                        base_latent_dim=16, use_derivatives=False).to(DEV)
ours.load_state_dict(torch.load(os.path.join(BASE, "best_raman2_model.pt"),
                                map_location=DEV, weights_only=True))
ours.eval()

casc = CascadedUNet().to(DEV)
casc.load_state_dict(torch.load(os.path.join(BASE, "runs", "kazemzadeh_seed0_best.pt"),
                                map_location=DEV, weights_only=True))
casc.eval()

R = {}
for c in CONCS:
    x, raw = load(c)
    lo, hi = float(raw.min()), float(raw.max())
    t = torch.tensor((raw - lo) / (hi - lo), dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        o = ours(t)
        f, _i, _b = casc(t)
    k = (hi - lo) / 10.0
    R[c] = dict(x=x, ref=airPLSz(raw, 1e4, 2, 40),
                ours=o[0].squeeze().numpy() * k,
                casc=f.squeeze().numpy() * k)


def amp(x, y, b, hw=8.0):
    w = np.abs(x - b) <= hw
    return float(y[w].max())


fig = plt.figure(figsize=(11.4, 7.8), dpi=150)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, 0])
ax_c = fig.add_subplot(gs[1, 1])


def tidy(a):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", color="0.93", lw=0.7)
    a.set_axisbelow(True)


# (a) full range, 200 mM
d = R[200]
m = (d["x"] >= 320) & (d["x"] <= 1700)
ax_a.plot(d["x"][m], d["ref"][m], color=C_REF, lw=0.9, label="airPLS (reference)", zorder=2)
ax_a.plot(d["x"][m], d["ours"][m], color=C_OURS, lw=1.7, label="dual-branch (this work)", zorder=4)
ax_a.plot(d["x"][m], d["casc"][m], color=C_CASC, lw=1.7, label="cascaded U-Net", zorder=3)
for b in (1115, 1162, 673):
    ax_a.axvline(b, color="0.75", lw=0.8, ls=":", zorder=1)
ax_a.set_xlim(320, 1700)
ax_a.set_xlabel("Raman shift (cm$^{-1}$)")
ax_a.set_ylabel("Recovered intensity (counts)")
ax_a.set_title("(a)  Recovered Raman signal, 200 mM glycerol", fontsize=10, loc="left")
ax_a.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
tidy(ax_a)

# (b) weak-band zoom
m2 = (d["x"] >= 1080) & (d["x"] <= 1220)
ax_b.plot(d["x"][m2], d["ref"][m2], color=C_REF, lw=1.0, label="airPLS", zorder=2)
ax_b.plot(d["x"][m2], d["ours"][m2], color=C_OURS, lw=2.0, label="dual-branch", zorder=4)
ax_b.plot(d["x"][m2], d["casc"][m2], color=C_CASC, lw=2.0, label="cascade", zorder=3)
for b, lab in ((1115, "1115"), (1162, "1162")):
    ax_b.axvline(b, color="0.75", lw=0.8, ls=":", zorder=1)
    ax_b.annotate(lab, xy=(b, 1.0), xycoords=("data", "axes fraction"),
                  xytext=(0, -11), textcoords="offset points",
                  ha="center", va="top", fontsize=8, color="0.3")
ax_b.set_xlim(1080, 1220)
ax_b.set_xlabel("Raman shift (cm$^{-1}$)")
ax_b.set_ylabel("Recovered intensity (counts)")
ax_b.set_title("(b)  Weak bands at 1115 and 1162 cm$^{-1}$, 200 mM", fontsize=10, loc="left")
ax_b.legend(frameon=False, fontsize=9)
tidy(ax_b)

# (c) recovered amplitude, band by band, 200 mM
o_amp = [amp(d["x"], d["ours"], b) for b in BANDS]
c_amp = [amp(d["x"], d["casc"], b) for b in BANDS]
idx = np.arange(len(BANDS))
w = 0.38
ax_c.bar(idx - w / 2, o_amp, w, color=C_OURS, label="dual-branch", zorder=3)
ax_c.bar(idx + w / 2, c_amp, w, color=C_CASC, label="cascade", zorder=3)
ax_c.set_xticks(idx)
ax_c.set_xticklabels([str(b) for b in BANDS], rotation=60, fontsize=8)
ax_c.set_xlabel("Reported band (cm$^{-1}$)")
ax_c.set_ylabel("Recovered amplitude (counts)")
ax_c.set_title("(c)  Recovered amplitude by band, 200 mM", fontsize=10, loc="left")
ax_c.legend(frameon=False, fontsize=9)
tidy(ax_c)

fig.tight_layout()
out = os.path.join(FIGS, "glycerol_cascade_comparison.png")
fig.savefig(out)
print(f"wrote {out}\n")

print(f"{'band':>6} " + "".join(f"{c:>16} mM" for c in CONCS))
print(f"{'':>6} " + "   ours  casc  ratio" * 3)
for b in BANDS:
    row = f"{b:>6} "
    for c in CONCS:
        dd = R[c]
        a1, a2 = amp(dd["x"], dd["ours"], b), amp(dd["x"], dd["casc"], b)
        row += f" {a1:6.0f}{a2:6.0f}{a1 / max(a2, 1e-9):6.2f}"
    print(row)
