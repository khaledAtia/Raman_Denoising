"""SI figure: the 1000-1100 cm-1 region of the tyrosine series.

A band near 1073 cm-1 is present in every tyrosine spectrum and does not grow with
concentration, which invites the question of whether the network is inventing it. The
figure answers that question per concentration, in two rows:

  top    the measurement as acquired, with the baseline the network predicts drawn over
         it -- the feature is an excursion of the raw data, before any correction
  bottom the same window after correction, three ways: airPLS (an estimator developed
         independently of this work), the raw spectrum minus the network's predicted
         baseline, and the Raman signal the network reports

If the feature were an artefact of the network it would be absent from the airPLS trace.

    python make_tyrosine_zoom.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(BASE, "paper_revised", "figures")
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, BASE)
from RamanUtils.airPLS import airPLSz

LO, HI = 1000.0, 1100.0
NOISE_WIN = (1700.0, 1880.0)
CONCS = (25, 50, 75, 100)
BAND = 1072.7
MARKER = 859.0                     # the tyrosine marker, for the scaling contrast
TMPL = os.path.join(BASE, "Spectrum", "Tyrosine", "{c}mM_Processed",
                    "Acquisition_s1_P.txt")

# concentration ramp of manuscript Figure 11, light (25 mM) to dark (100 mM)
RAMP = {25: "#bdd7e7", 50: "#6baed6", 75: "#2171b5", 100: "#08306b"}
C_AIRPLS = "#6b6b6b"
C_BASE = "#d94801"
SG_WIN, SG_ORDER = 9, 3


def sg(y):
    return savgol_filter(y, SG_WIN, SG_ORDER)


def sigma_of(x, y):
    m = (x >= NOISE_WIN[0]) & (x <= NOISE_WIN[1])
    d = np.diff(y[m])
    return np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2)


def load(c):
    d = np.loadtxt(TMPL.format(c=c), skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def dress(a, title):
    a.set_xlim(LO, HI)
    a.set_title(title, fontsize=9.5, loc="left")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", color="0.92", lw=0.7, zorder=0)
    a.set_axisbelow(True)
    a.axvline(BAND, color="0.45", lw=0.9, ls="--", zorder=2)


fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.6), dpi=150)
report = []

for k, c in enumerate(CONCS):
    x, raw, pr, pb = load(c)
    ap = airPLSz(raw, 1e4, 2, 40)
    resid = raw - pb
    m = (x >= LO) & (x <= HI)
    s_ap = sigma_of(x, ap)

    # ---- top row: the measurement as acquired, with the predicted baseline -----
    ax = axes[0, k]
    ax.plot(x[m], raw[m], color=RAMP[c], lw=1.6, zorder=3, label="measured")
    ax.plot(x[m], pb[m], color=C_BASE, lw=1.4, ls="--", zorder=4,
            label="predicted baseline")
    dress(ax, f"({chr(97+k)})  {c}~mM, as acquired".replace("~", " "))
    ax.set_ylabel("Intensity (counts)" if k == 0 else "")
    if k == 0:
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    # ---- bottom row: corrected three ways -------------------------------------
    ax = axes[1, k]
    ax.axhspan(-3 * s_ap, 3 * s_ap, color="0.93", zorder=0, lw=0)
    ax.axhline(0.0, color="0.55", lw=0.8, zorder=1)
    ax.plot(x[m], ap[m], color=C_AIRPLS, lw=0.7, alpha=0.35, zorder=2)
    ax.plot(x[m], sg(ap)[m], color=C_AIRPLS, lw=1.5, zorder=3, label="airPLS")
    ax.plot(x[m], sg(resid)[m], color=C_BASE, lw=1.3, ls=(0, (4, 2)), zorder=4,
            label="raw $-$ predicted baseline")
    ax.plot(x[m], pr[m], color=RAMP[c], lw=2.0, zorder=5, label="network output")
    dress(ax, f"({chr(101+k)})  {c} mM, baseline removed")
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (counts)" if k == 0 else "")
    if k == 0:
        ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    w = (x >= 1040) & (x <= 1100)
    wm = np.abs(x - MARKER) <= 8
    report.append((c, x[w][pr[w].argmax()], ap[w].max(), resid[w].max(),
                   pr[w].max(), s_ap, pr[wm].max()))

# common y-limits per row so the four concentrations are directly comparable
for r in range(2):
    lo = min(a.get_ylim()[0] for a in axes[r])
    hi = max(a.get_ylim()[1] for a in axes[r])
    if r == 0:
        hi += 0.10 * (hi - lo)          # headroom above the measured trace
    for a in axes[r]:
        a.set_ylim(lo, hi)

for k in range(4):
    for r in range(2):
        axes[r, k].annotate(f"{BAND:.0f}", xy=(BAND, 1.0),
                            xycoords=("data", "axes fraction"),
                            xytext=(3, -11), textcoords="offset points",
                            ha="left", va="top", fontsize=8, color="0.25")

fig.tight_layout()
out = os.path.join(FIGS, "tyrosine_zoom_1000_1100.png")
fig.savefig(out)
print(f"wrote {out}\n")

print(f"{'conc':>5} {'position':>9} {'airPLS':>9} {'raw-predB':>10} {'network':>9}"
      f" {'sigma':>7} {'SNR':>6} | {'859 marker':>11}")
for c, pos, a, r, n, sd, mk in report:
    print(f"{c:>5} {pos:>9.1f} {a:>9.1f} {r:>10.1f} {n:>9.1f} {sd:>7.1f}"
          f" {a/sd:>6.1f} | {mk:>11.1f}")

amps = np.array([r[2] for r in report])
mks = np.array([r[6] for r in report])
print(f"\n1073 band, airPLS:  100/25 ratio {amps[-1]/amps[0]:.2f}"
      f"   (spread {amps.min():.0f}-{amps.max():.0f} counts)")
print(f"859 marker, network: 100/25 ratio {mks[-1]/mks[0]:.2f}")
