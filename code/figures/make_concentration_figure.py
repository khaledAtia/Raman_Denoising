"""Figure 10(b) rebuilt with replicate error bars, and compared with the published panel.

Plots the deep-layer photon count against guanine concentration as mean +/- 1 SD over the
five spectra recorded at each concentration, with the regression taken through the means.
A second panel overlays the published Figure 10(b) response, each series normalised to its
own 80 mM value, so the shapes can be compared independently of overall brightness.

    python make_concentration_figure.py [root]
"""

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from Pmodel import AUSequentialUNet

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "Spectrum",
                    sys.argv[1] if len(sys.argv) > 1 else "guanine3/guanine2_concentrations")
TAG = os.path.basename(ROOT.rstrip("/\\"))
FIG = os.path.join(BASE, f"concentration_response_{TAG}.png")
OUT = os.path.join(ROOT, "photon_count_report.txt")

# Figure 10(b) of the manuscript (conc_vs_photons in paper_validate_guanine.py)
PAPER = {20: 54773.02182417218, 30: 207249.20876429786, 40: 337477.9675843251,
         50: 478713.6930251343, 60: 621013.41382320493, 70: 750800.6725386118,
         80: 917925.4839693243}

C_PT, C_FIT = "#1f77b4", "#ff7f0e"
C_PAPER = "#7f7f7f"


def load_model():
    m = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0, kernels=[1, 1, 3],
                         base_latent_dim=16, use_derivatives=False)
    m.load_state_dict(torch.load(os.path.join(BASE, "best_raman2_model.pt"),
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


def i_deep(model, raw):
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    t = torch.tensor((raw - lo) / span, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        o = model(t)
    return span / 10.0 * float(o[3].squeeze().numpy().sum())


def fit(x, y):
    r = stats.linregress(x, y)
    n = len(x)
    tc = stats.t.ppf(0.975, n - 2)
    resid = np.asarray(y) - (r.slope * np.asarray(x) + r.intercept)
    rse = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    return r, tc, rse


def main():
    model = load_model()
    vals = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "*mM_LaserRemoved")),
                    key=lambda p: int(re.search(r"(\d+)mM", os.path.basename(p)).group(1))):
        c = int(re.search(r"(\d+)mM", os.path.basename(d)).group(1))
        vs = []
        for f in sorted(glob.glob(os.path.join(d, "*.asc"))):
            a = np.loadtxt(f)
            vs += [i_deep(model, a[:, i]) for i in range(1, a.shape[1]) if a[:, i].max() > 0]
        vals[c] = vs

    cs = np.array(sorted(vals))
    mu = np.array([np.mean(vals[c]) for c in cs])
    sd = np.array([np.std(vals[c], ddof=1) for c in cs])
    pv = np.array([PAPER[c] for c in cs])

    fm, tcm, rsem = fit(cs, mu)
    fp, tcp, rsep = fit(cs, pv)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=170)

    ax = axes[0]
    xx = np.linspace(cs.min() - 4, cs.max() + 4, 200)
    se = rsem * np.sqrt(1 / len(cs) + (xx - cs.mean()) ** 2 / ((cs - cs.mean()) ** 2).sum())
    yy = fm.slope * xx + fm.intercept
    ax.fill_between(xx, yy - tcm * se, yy + tcm * se, color=C_FIT, alpha=0.14, lw=0, zorder=2)
    ax.plot(xx, yy, color=C_FIT, ls="--", lw=1.5, zorder=3, label="Linear Fit")
    ax.errorbar(cs, mu, yerr=sd, fmt="o", ms=6, color=C_PT, capsize=4, lw=1.4,
                mfc="white", mew=1.7, zorder=4,
                label="Deep layer photons (mean $\\pm$ 1 SD, $n$ = 5)")
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Photon Count")
    ax.set_xticks(cs)
    ax.set_xlim(cs.min() - 5, cs.max() + 5)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.text(0.05, 0.94, f"$R^2 = {fm.rvalue**2:.4f}$", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("(a)  This measurement, five spectra per concentration",
                 fontsize=10, loc="left")

    ax = axes[1]
    mn, pn = mu / mu[-1], pv / pv[-1]
    sn = sd / mu[-1]
    ax.errorbar(cs, mn, yerr=sn, fmt="o", ms=6, color=C_PT, capsize=4, lw=1.4,
                mfc="white", mew=1.7, zorder=4, label="This measurement")
    ax.plot(cs, pn, "s--", ms=5, color=C_PAPER, lw=1.4, zorder=3,
            label="Published Figure 10(b)")
    ax.set_xlabel("Concentration (mM)")
    ax.set_ylabel("Photon count, normalised to 80 mM")
    ax.set_xticks(cs)
    ax.set_xlim(cs.min() - 5, cs.max() + 5)
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_title("(b)  Shape of the response, scale removed", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(FIG)

    # ---------------------------------------------------------------- report
    L = ["Deep-layer photon count vs guanine concentration -- " + TAG,
         "=" * 100, "",
         "I_deep = (X_max - X_min)/10 * sum of the 216-point deep auxiliary head.",
         "'paper' is Figure 10(b) of the manuscript at the same concentration.", "",
         "-" * 100,
         f"{'conc_mM':>9}{'measured_mean':>17}{'SD':>13}{'RSD_%':>9}{'paper':>15}"
         f"{'measured/paper':>17}",
         "-" * 100]
    for c, m_, s_, p_ in zip(cs, mu, sd, pv):
        L.append(f"{c:9d}{m_:17.1f}{s_:13.1f}{100*s_/m_:9.2f}{p_:15.1f}{m_/p_:17.2f}")
    L += ["-" * 100, "",
          "Linear response", "-" * 100,
          f"{'':22}{'slope (per mM)':>18}{'intercept':>16}{'R^2':>10}{'residual SE':>15}",
          f"{'measured':22}{fm.slope:18.1f}{fm.intercept:16.1f}{fm.rvalue**2:10.4f}{rsem:15.1f}",
          f"{'paper, Figure 10(b)':22}{fp.slope:18.1f}{fp.intercept:16.1f}"
          f"{fp.rvalue**2:10.4f}{rsep:15.1f}",
          "",
          f"measured slope 95% CI:     {fm.slope:.1f} +/- {tcm*fm.stderr:.1f}",
          f"measured intercept 95% CI: {fm.intercept:.1f} +/- {tcm*fm.intercept_stderr:.1f}",
          f"slope ratio measured/paper: {fm.slope/fp.slope:.3f}",
          "-" * 100, "",
          "Shape of the response, each series normalised to its own 80 mM value",
          "-" * 100,
          f"{'conc_mM':>9}{'measured':>14}{'paper':>13}{'difference_%':>16}",
          "-" * 100]
    for c, a, b in zip(cs, mn, pn):
        L.append(f"{c:9d}{a:14.4f}{b:13.4f}{100*(a-b)/b:16.1f}")
    L += ["-" * 100,
          f"mean relative difference in shape: {100*np.abs((mn-pn)/pn).mean():.1f}%",
          "-" * 100]

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {FIG}\nwrote {OUT}")


if __name__ == "__main__":
    main()
