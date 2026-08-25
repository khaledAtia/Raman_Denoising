"""SI figure: the 550-760 cm-1 region of the glycerol series used for Figure 8.

Four views of the same window, for the three measured concentrations:

  (a) baseline-corrected with airPLS, an estimator independent of this work
  (b) raw spectrum minus the baseline predicted by our network
  (c) the Raman signal reported by our network
  (d) the analyte-free microplate background

The dataset is Spectrum/Glycerin2, which is the series Figure 8 of the manuscript
was produced from; the released checkpoint reproduces its stored predictions to
better than 0.2% of peak.

    python make_glycerol_zoom.py
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

LO, HI = 550.0, 760.0
NOISE_WIN = (1700.0, 1880.0)
CONCS = (100, 150, 200)
MARKS = (616.0, 673.0)
TMPL = os.path.join(BASE, "Spectrum", "Glycerin2", "{c}_P.txt")

RAMP = {100: "#9ecae1", 150: "#4292c6", 200: "#08519c"}
BG_COLOR = "#6b6b6b"

# Savitzky-Golay smoothing of the baseline-corrected traces, for display only.
# The window is held just below the narrowest band in the window: the features
# here have a FWHM near 19 cm-1 and the 852 cm-1 band 16 cm-1, against a window
# of 9 points = 16.5 cm-1 at the 1.833 cm-1/pt dispersion of these spectra. On a
# noise-free Lorentzian of those widths the filter costs 2.2% and 3.7% of peak
# amplitude respectively. Every number quoted in the text and in the band table
# is measured on the UNSMOOTHED trace; smoothing is never applied before
# quantification.
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


def dress(a, title, marks=True):
    a.set_xlim(LO, HI)
    a.set_xlabel("Raman shift (cm$^{-1}$)")
    a.set_ylabel("Intensity (counts)")
    a.set_title(title, fontsize=10, loc="left")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", color="0.92", lw=0.7, zorder=0)
    a.set_axisbelow(True)
    if marks:
        for pos in MARKS:
            a.axvline(pos, color="0.45", lw=0.9, ls="--", zorder=2)


fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), dpi=150)
ax = axes.ravel()

# ------------------------------------------------- (a) independent baseline
sig_a = []
for c in CONCS:
    x, raw, _, _ = load(c)
    corr = airPLSz(raw, 1e4, 2, 40)
    sig_a.append(sigma_of(x, corr))
    m = (x >= LO) & (x <= HI)
    ax[0].plot(x[m], corr[m], color=RAMP[c], lw=0.7, alpha=0.30, zorder=2)
    ax[0].plot(x[m], sg(corr)[m], color=RAMP[c], lw=1.7, label=f"{c} mM", zorder=3)
sa = float(np.mean(sig_a))
ax[0].axhspan(-3 * sa, 3 * sa, color="0.93", zorder=0, lw=0)
for s_ in (-3 * sa, 3 * sa):
    ax[0].axhline(s_, color="0.62", lw=0.8, ls=":", zorder=1)
ax[0].axhline(0.0, color="0.55", lw=0.8, zorder=1)
dress(ax[0], "(a)  airPLS baseline correction (independent of this work)")
ax[0].legend(frameon=False, fontsize=9, loc="upper right")
ax[0].annotate(f"$\\pm3\\sigma$, $\\sigma\\approx{sa:.0f}$ counts (unsmoothed)\n"
               f"bold: Savitzky--Golay, {SG_WIN} pt, order {SG_ORDER}; faint: as corrected",
               xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color="0.35")

# ------------------------------------------- (b) our model's baseline removed
sig_b = []
for c in CONCS:
    x, raw, _, pb = load(c)
    resid = raw - pb
    sig_b.append(sigma_of(x, resid))
    m = (x >= LO) & (x <= HI)
    ax[1].plot(x[m], resid[m], color=RAMP[c], lw=0.7, alpha=0.30, zorder=2)
    ax[1].plot(x[m], sg(resid)[m], color=RAMP[c], lw=1.7, label=f"{c} mM", zorder=3)
sb = float(np.mean(sig_b))
ax[1].axhspan(-3 * sb, 3 * sb, color="0.93", zorder=0, lw=0)
for s_ in (-3 * sb, 3 * sb):
    ax[1].axhline(s_, color="0.62", lw=0.8, ls=":", zorder=1)
ax[1].axhline(0.0, color="0.55", lw=0.8, zorder=1)
dress(ax[1], "(b)  raw $-$ baseline predicted by our network")
ax[1].legend(frameon=False, fontsize=9, loc="upper right")
ax[1].annotate(f"$\\pm3\\sigma$, $\\sigma\\approx{sb:.0f}$ counts (unsmoothed)\n"
               f"bold: Savitzky--Golay, {SG_WIN} pt, order {SG_ORDER}; faint: as corrected",
               xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color="0.35")

# ------------------------------------------------ (c) network Raman output
for c in CONCS:
    x, _, pr, _ = load(c)
    m = (x >= LO) & (x <= HI)
    ax[2].plot(x[m], pr[m], color=RAMP[c], lw=1.9, label=f"{c} mM", zorder=3)
ax[2].axhline(0.0, color="0.55", lw=0.8, zorder=1)
dress(ax[2], "(c)  Raman signal reported by our network")
ax[2].legend(frameon=False, fontsize=9, loc="upper right")
ax[2].annotate("one feature is reported at a time",
               xy=(0.03, 0.72), xycoords="axes fraction", fontsize=8.5, color="0.3")

# --------------------------------------------------- (d) analyte-free bank
bank = np.load(os.path.join(BASE, "data", "baseline_data.npz"))["data"]
xb, spectra = bank[:, 0], bank[:, 1:]
bg = airPLSz(spectra.mean(axis=1), 1e4, 2, 40)
sig_bg = sigma_of(xb, bg)
mb = (xb >= LO) & (xb <= HI)
ax[3].plot(xb[mb], bg[mb], color=BG_COLOR, lw=1.6, zorder=3,
           label=f"mean of {spectra.shape[1]} wells")
ax[3].axhspan(-3 * sig_bg, 3 * sig_bg, color="0.93", zorder=0, lw=0)
ax[3].axhline(0.0, color="0.55", lw=0.8, zorder=1)
dress(ax[3], "(d)  Analyte-free background, water-filled wells")
ax[3].legend(frameon=False, fontsize=9, loc="upper right")
ax[3].annotate(f"note the scale: peak $\\approx${bg[mb].max():.0f} counts",
               xy=(0.02, 0.04), xycoords="axes fraction", fontsize=8, color="0.35")

for a in ax:
    a.annotate("616", xy=(616.0, 1.0), xycoords=("data", "axes fraction"),
               xytext=(-2, -11), textcoords="offset points",
               ha="right", va="top", fontsize=8, color="0.25")
    a.annotate("673", xy=(673.0, 1.0), xycoords=("data", "axes fraction"),
               xytext=(2, -11), textcoords="offset points",
               ha="left", va="top", fontsize=8, color="0.25")

fig.tight_layout()
out = os.path.join(FIGS, "glycerol_zoom_600_700.png")
fig.savefig(out)
print(f"wrote {out}\n")

print(f"sigma: airPLS {sa:.1f}, our-baseline residual {sb:.1f}, background {sig_bg:.2f}\n")
print(f"{'conc':>6} {'616 meas':>10} {'616 SNR':>9} {'616 model':>10}"
      f" {'673 meas':>10} {'673 SNR':>9} {'673 model':>10}")
for c, s in zip(CONCS, sig_a):
    x, raw, pr, _ = load(c)
    corr = airPLSz(raw, 1e4, 2, 40)
    row = f"{c:>6}"
    for b in (616, 673):
        w = np.abs(x - b) <= 8
        row += f" {corr[w].max():>10.1f} {corr[w].max()/s:>9.2f} {pr[w].max():>10.1f}"
    print(row)

# The figure smooths for display only. Confirm the conclusion of this section --
# 673 grows with concentration, 616 does not -- is a property of the data and not
# of the filter, by re-deriving the two ratios from the smoothed traces.
print("\nrobustness of the 200/100 amplitude ratio to the display filter:")
amp = {}
for c in CONCS:
    x, raw, _, _ = load(c)
    corr = airPLSz(raw, 1e4, 2, 40)
    for tag, y in (("raw", corr), ("sg", sg(corr))):
        for b in (616, 673):
            w = np.abs(x - b) <= 8
            amp[(tag, b, c)] = y[w].max()
for b in (616, 673):
    r_raw = amp[("raw", b, 200)] / amp[("raw", b, 100)]
    r_sg = amp[("sg", b, 200)] / amp[("sg", b, 100)]
    print(f"  {b} cm-1:  unsmoothed {r_raw:.2f}   smoothed {r_sg:.2f}")
