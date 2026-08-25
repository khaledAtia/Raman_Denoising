"""Visual confirmation that the calibrated axis puts the new guanine bands where the
paper reports them."""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from RamanUtils.airPLS import airPLS

BASE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = [654, 1165, 1200, 1230, 1270, 1335, 1380, 1460, 1540]
CAL = os.path.join(BASE, "NewData", "Guanine_Calibrated")
OWN = os.path.join(BASE, "NewData", "Guanine_LaserRemoved")


def corrected(path):
    d = np.loadtxt(path)
    cols = [d[:, i] for i in range(1, d.shape[1]) if d[:, i].max() > 0]
    y = np.mean(cols, axis=0)
    return d[:, 0], y - airPLS(y, 1e4, 2, 40)


def peaks_of(y, n=25):
    p, pr = find_peaks(y, prominence=(y.max() - y.min()) * 0.02, distance=4)
    return sorted(p[np.argsort(pr["prominences"])[::-1][:n]])


# aggregate over the high-signal subset, where peak finding is reliable
subset = [f for f in sorted(glob.glob(os.path.join(CAL, "*.asc")))
          if any(f"{c} mM" in f or f"{c}mM" in f for c in (40, 50, 70, 80))
          and ("50s" in f or "60s" in f)]
devs = []
for f in subset:
    ax, y = corrected(f)
    pk = ax[peaks_of(y)]
    devs += [min(pk, key=lambda p: abs(p - r)) - r for r in REFERENCE]
devs = np.array(devs)
print(f"high-signal subset: {len(subset)} files x {len(REFERENCE)} bands = {devs.size} matches")
print(f"  mean deviation {devs.mean():+.2f} cm-1, mean|dev| {np.abs(devs).mean():.2f}, "
      f"rms {np.sqrt((devs**2).mean()):.2f}, max|dev| {np.abs(devs).max():.2f}")
print(f"  within one 1.8 cm-1 sample: {(np.abs(devs) <= 1.8).mean()*100:.0f}% of bands")
print(f"  within two samples:         {(np.abs(devs) <= 3.7).mean()*100:.0f}% of bands")

name = "guanine 80 mM 60s.asc"
ax_c, y_c = corrected(os.path.join(CAL, name))
ax_o, y_o = corrected(os.path.join(OWN, name))

fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), dpi=150, sharey=True)
for a, x, y, ttl, col in (
        (axes[0], ax_c, y_c, "(a)  calibrated axis (data_calibrated.asc)", "#08519c"),
        (axes[1], ax_o, y_o, "(b)  axis as shipped inside the .asc files", "#d95f02")):
    a.plot(x, y, color=col, lw=1.2, zorder=3)
    for r in REFERENCE:
        a.axvline(r, color="0.65", lw=0.9, ls=":", zorder=1)
    a.set_xlim(400, 1700)
    a.set_title(ttl + f"   —  {name}", fontsize=10, loc="left")
    a.set_ylabel("Baseline-corrected counts")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
for r in REFERENCE:
    axes[0].annotate(str(r), xy=(r, 1.0), xycoords=("data", "axes fraction"),
                     xytext=(0, 2), textcoords="offset points",
                     ha="center", fontsize=7.5, color="0.35")
fig.suptitle("Dotted lines: guanine bands as reported in the manuscript", fontsize=9.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(BASE, "axis_calibration_check.png")
fig.savefig(out)
print(f"\nwrote {out}")
