"""Figure 10(d) rebuilt with replicate error bars.

Plots the deep-layer photon count against acquisition time for the guanine series,
as mean +/- 1 SD over the five spectra recorded at each exposure, with the linear
regression taken through those means.

    python make_photon_count_figure.py            # 100mM_2
    python make_photon_count_figure.py 100mM      # any processed folder
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
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "100mM_2"
SRC = os.path.join(BASE, "Spectrum", "Guanine", f"{FOLDER}_LaserRemoved")
OUT = os.path.join(BASE, f"photon_count_{FOLDER}.png")

C_PT, C_FIT, C_BAND = "#1f77b4", "#d62728", "#d62728"


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


def main():
    model = load_model()
    per_time = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.asc"))):
        t = int(re.search(r"(\d+)s", os.path.basename(f)).group(1))
        d = np.loadtxt(f)
        per_time[t] = [i_deep(model, d[:, i]) for i in range(1, d.shape[1])
                       if d[:, i].max() > 0]

    ts = np.array(sorted(per_time))
    mu = np.array([np.mean(per_time[t]) for t in ts])
    sd = np.array([np.std(per_time[t], ddof=1) for t in ts])
    n = len(ts)

    fit = stats.linregress(ts, mu)
    tcrit = stats.t.ppf(0.975, n - 2)
    resid = mu - (fit.slope * ts + fit.intercept)
    rse = np.sqrt((resid ** 2).sum() / (n - 2))

    xx = np.linspace(ts.min() - 4, ts.max() + 4, 200)
    yy = fit.slope * xx + fit.intercept
    se = rse * np.sqrt(1 / n + (xx - ts.mean()) ** 2 / ((ts - ts.mean()) ** 2).sum())

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=170)
    ax.fill_between(xx, yy - tcrit * se, yy + tcrit * se, color=C_BAND, alpha=0.12,
                    lw=0, zorder=2, label="95% CI of the fit")
    ax.plot(xx, yy, color=C_FIT, lw=1.6, zorder=3, label="Linear fit")
    ax.errorbar(ts, mu, yerr=sd, fmt="o", ms=6, color=C_PT, capsize=4, lw=1.4,
                mfc="white", mew=1.7, zorder=4,
                label="Deep layer photons (mean $\\pm$ 1 SD, $n$ = 5)")

    ax.set_xlabel("Acquisition Time (s)")
    ax.set_ylabel("Photon Count")
    ax.set_xlim(ts.min() - 4, ts.max() + 4)
    ax.set_xticks(ts)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    ax.annotate(f"$R^2$ = {fit.rvalue**2:.4f}\n"
                f"$I$ = {fit.slope:,.0f}$\\,t$ + {fit.intercept:,.0f}",
                xy=(0.035, 0.965), xycoords="axes fraction", va="top", ha="left",
                fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.92", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT)

    print(f"{FOLDER}: {n} exposures, 5 spectra each")
    print(f"{'time_s':>8}{'mean':>16}{'SD':>14}{'RSD_%':>9}")
    for t, m_, s_ in zip(ts, mu, sd):
        print(f"{t:8d}{m_:16.1f}{s_:14.1f}{100*s_/m_:9.2f}")
    print(f"\nslope     {fit.slope:,.0f} +/- {tcrit*fit.stderr:,.0f} per s (95% CI)")
    print(f"intercept {fit.intercept:,.0f} +/- {tcrit*fit.intercept_stderr:,.0f} (95% CI)")
    print(f"R2 {fit.rvalue**2:.4f}   residual SE {rse:,.0f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
