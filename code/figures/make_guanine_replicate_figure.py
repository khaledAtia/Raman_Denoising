"""Figure 10(b) and 10(d) rebuilt on the recalibrated guanine series, with error bars.

Each .asc in NewData/Guanine_Calibrated holds five spectra of one condition, so every
point carries a dispersion where the published figure had a single measurement. The
readout is the one defined in the manuscript,

    I_deep = (X_max - X_min)/10 * sum_i s_deep_i ,

computed through the unchanged trained model and the same inference path as
make_concentration_response.py.

    python make_guanine_replicate_figure.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from Pmodel import AUSequentialUNet

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "NewData", "Guanine_Calibrated")
CONC_AT_TIME = 60      # panel (a): concentration series at this acquisition time
TIME_AT_CONC = 80      # panel (b): acquisition-time series at this concentration
C_PT, C_FIT = "#08519c", "#d95f02"


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


def measure(model):
    """{(conc, time): [I_deep per valid spectrum]}, plus notes on what was dropped."""
    out, notes = {}, []
    for f in sorted(glob.glob(os.path.join(DATA, "*.asc"))):
        name = os.path.basename(f)
        m = re.search(r"(\d+)\s*mM\s*(\d+)\s*s", name)
        conc, time = int(m.group(1)), int(m.group(2))
        d = np.loadtxt(f)
        vals = []
        for i in range(1, d.shape[1]):
            y = d[:, i]
            if y.max() <= 0:
                notes.append(f"{name}: spectrum {i} is all zero, excluded")
                continue
            if (y >= 65535).any():
                notes.append(f"{name}: spectrum {i} is saturated (cosmic ray), excluded")
                continue
            vals.append(i_deep(model, y))
        out[(conc, time)] = vals
    return out, notes


def regress(x, y):
    """Least squares on the individual replicate measurements, with 95% intervals."""
    r = stats.linregress(x, y)
    n = len(x)
    tcrit = stats.t.ppf(0.975, n - 2)
    resid = np.asarray(y) - (r.slope * np.asarray(x) + r.intercept)
    rse = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    return dict(slope=r.slope, slope_ci=tcrit * r.stderr,
                intercept=r.intercept, intercept_ci=tcrit * r.intercept_stderr,
                r2=r.rvalue ** 2, rse=rse, n=n,
                lod=3 * rse / r.slope, loq=10 * rse / r.slope)


def panel(ax, xs, groups, fit, xlabel, title, unit):
    mu = np.array([np.mean(groups[k]) for k in xs])
    sd = np.array([np.std(groups[k], ddof=1) for k in xs])
    ax.errorbar(xs, mu, yerr=sd, fmt="o", ms=5.5, color=C_PT, capsize=3.5,
                lw=1.3, mfc="white", mew=1.5, zorder=4,
                label="mean $\\pm$ 1 SD of 5 accumulations")
    xx = np.linspace(min(xs) * 0.94, max(xs) * 1.06, 100)
    fm = regress(list(xs), list(mu))          # fit through the condition means
    ax.plot(xx, fm["slope"] * xx + fm["intercept"], ls="--", lw=1.4,
            color=C_FIT, zorder=3,
            label=f"fit to means, $R^2$ = {fm['r2']:.4f}")
    fit = fm
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$I_{\\mathrm{deep}}$ (counts)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="0.93", lw=0.7)
    ax.set_axisbelow(True)
    rsd = 100 * sd / mu
    print(f"\n  {title}")
    print(f"    {unit:>8}{'mean':>12}{'SD':>10}{'RSD %':>8}{'n':>4}")
    for k, m_, s_, r_ in zip(xs, mu, sd, rsd):
        print(f"    {k:>8}{m_:12.1f}{s_:10.1f}{r_:8.2f}{len(groups[k]):4d}")
    print(f"    slope {fit['slope']:.3f} +/- {fit['slope_ci']:.3f}   "
          f"intercept {fit['intercept']:.1f} +/- {fit['intercept_ci']:.1f}")
    print(f"    R2 {fit['r2']:.4f}   residual SE {fit['rse']:.1f}   n = {fit['n']} spectra")
    print(f"    LOD (3 sigma/slope) {fit['lod']:.2f}   LOQ (10 sigma/slope) {fit['loq']:.2f}")


def main():
    model = load_model()
    data, notes = measure(model)
    for n in notes:
        print("NOTE", n)

    concs = sorted({c for c, t in data if t == CONC_AT_TIME})
    times = sorted({t for c, t in data if c == TIME_AT_CONC})

    g_c = {c: data[(c, CONC_AT_TIME)] for c in concs}
    g_t = {t: data[(TIME_AT_CONC, t)] for t in times}

    fc = regress([c for c in concs for _ in g_c[c]], [v for c in concs for v in g_c[c]])
    ft = regress([t for t in times for _ in g_t[t]], [v for t in times for v in g_t[t]])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), dpi=150)
    panel(axes[0], concs, g_c, fc, "Guanine concentration (mM)",
          f"(a)  Concentration series, {CONC_AT_TIME} s acquisition", "mM")
    panel(axes[1], times, g_t, ft, "Acquisition time (s)",
          f"(b)  Acquisition-time series, {TIME_AT_CONC} mM", "s")
    fig.tight_layout()
    out = os.path.join(BASE, "guanine_replicates.png")
    fig.savefig(out)
    print(f"\nwrote {out}")

    print("\nfull grid, mean I_deep (SD) over valid spectra:")
    allc = sorted({c for c, _ in data})
    allt = sorted({t for _, t in data})
    print("      " + "".join(f"{t:>16}s" for t in allt))
    for c in allc:
        row = f"{c:>4}mM "
        for t in allt:
            v = data.get((c, t))
            row += f"{np.mean(v):9.0f} ({np.std(v, ddof=1):5.0f})" if v else f"{'--':>17}"
        print(row)


if __name__ == "__main__":
    main()
