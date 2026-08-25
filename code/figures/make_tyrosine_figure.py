"""SI figure for the independent analyte: tyrosine in 1 M NaOH.

Four panels, as Section S10.1 requires: the raw measurement with the predicted baseline,
the recovered Raman signal against the difference signal that is independent of the Raman
branch, the recovered signal across the concentration series, and the integrated response.

    python make_tyrosine_figure.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "Spectrum", "Tyrosine")
OUT = os.path.join(BASE, "paper_revised", "figures", "tyrosine_validation.png")

BANDS = [649, 837, 859, 1179, 1214, 1275, 1450, 1608]
LO, HI = 400, 1700
C_RAW, C_BASE, C_SIG, C_DIFF = "0.55", "#2ca02c", "#8c564b", "#9467bd"


def load():
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "*mM_Processed")),
                    key=lambda p: int(os.path.basename(p).split("mM")[0])):
        c = int(os.path.basename(d).split("mM")[0])
        a = np.loadtxt(glob.glob(os.path.join(d, "*.txt"))[0], skiprows=1)
        out[c] = dict(x=a[:, 0], raw=a[:, 1], sig=a[:, 2], base=a[:, 3])
    return out


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.93", lw=0.7)
    ax.set_axisbelow(True)


def main():
    D = load()
    concs = sorted(D)
    top = D[concs[-1]]
    m = (top["x"] >= LO) & (top["x"] <= HI)

    fig = plt.figure(figsize=(11.2, 7.4), dpi=170)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

    # (a) raw and predicted baseline
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(top["x"][m], top["raw"][m], color=C_RAW, lw=1.0, label="Raw spectrum", zorder=2)
    ax.plot(top["x"][m], top["base"][m], color=C_BASE, lw=1.6, ls="--",
            label="Predicted baseline", zorder=3)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"(a)  Measurement and predicted baseline, {concs[-1]} mM",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)

    # (b) recovered signal against the difference signal
    ax = fig.add_subplot(gs[0, 1])
    diff = top["raw"] - top["base"]
    for b in BANDS:
        ax.axvspan(b - 10, b + 10, color="gray", alpha=0.18, zorder=0)
    ax.plot(top["x"][m], diff[m], color=C_DIFF, lw=1.0, alpha=0.75,
            label="Raw $-$ predicted baseline", zorder=2)
    ax.plot(top["x"][m], top["sig"][m], color=C_SIG, lw=1.6,
            label="Recovered Raman signal", zorder=3)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"(b)  Recovered signal against the independent difference, "
                 f"{concs[-1]} mM", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    tidy(ax)

    # (c) the concentration series
    ax = fig.add_subplot(gs[1, 0])
    for i, c in enumerate(concs):
        d = D[c]
        mm = (d["x"] >= LO) & (d["x"] <= HI)
        ax.plot(d["x"][mm], d["sig"][mm], lw=1.3,
                color=plt.cm.viridis(i / max(len(concs) - 1, 1)), label=f"{c} mM")
    for b in BANDS:
        ax.axvspan(b - 10, b + 10, color="gray", alpha=0.15, zorder=0)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Recovered intensity (counts)")
    ax.set_title("(c)  Recovered Raman signal across the series", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    tidy(ax)

    # (d) integrated response
    ax = fig.add_subplot(gs[1, 1])
    xs = np.array(concs, float)
    ys = np.array([D[c]["sig"][(D[c]["x"] >= LO) & (D[c]["x"] <= HI)].sum() for c in concs])
    r = stats.linregress(xs, ys)
    xx = np.linspace(xs.min() - 8, xs.max() + 8, 100)
    ax.plot(xx, r.slope * xx + r.intercept, ls="--", lw=1.4, color="#d95f02",
            label="Linear fit", zorder=2)
    ax.plot(xs, ys, "o", ms=7, color="#08519c", mfc="white", mew=1.8, zorder=3,
            label="Integrated recovered signal")
    ax.set_xlabel("Tyrosine concentration (mM)")
    ax.set_ylabel("Integrated recovered signal (counts)")
    ax.set_xticks(xs)
    ax.set_title("(d)  Integrated response against concentration", fontsize=10, loc="left")
    ax.annotate(f"$R^2$ = {r.rvalue**2:.4f}", xy=(0.05, 0.93), xycoords="axes fraction",
                fontsize=10, va="top")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(ax)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"integrated response: R2 = {r.rvalue**2:.4f}, "
          f"slope {r.slope:,.0f} per mM, intercept {r.intercept:,.0f}")
    print(f"{'conc':>8}{'integral':>16}{'peak':>12}")
    for c, y in zip(concs, ys):
        d = D[c]
        mm = (d["x"] >= LO) & (d["x"] <= HI)
        print(f"{c:>6} mM{y:16.0f}{d['sig'][mm].max():12.0f}")


if __name__ == "__main__":
    main()
