"""Deep-layer photon count for the guanine concentration series, against Figure 10(b).

For every spectrum under Spectrum/guanine_concentrations/<conc>mM_LaserRemoved this
computes the manuscript's readout,

    I_deep = (X_max - X_min)/10 * sum_i s_deep_i ,

and places it beside the value Figure 10(b) reports at the same concentration. The
published values are those hard-coded in paper_validate_guanine.py.

    python report_concentration_counts.py
"""

import glob
import os
import re

import numpy as np
import torch
from scipy import stats

from Pmodel import AUSequentialUNet

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "Spectrum", "guanine_concentrations")
OUT = os.path.join(ROOT, "photon_count_report.txt")

# Figure 10(b) of the manuscript (conc_vs_photons in paper_validate_guanine.py)
PAPER = {20: 54773.02182417218,
         30: 207249.20876429786,
         40: 337477.9675843251,
         50: 478713.6930251343,
         60: 621013.41382320493,
         70: 750800.6725386118,
         80: 917925.4839693243}


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


def fit_stats(x, y):
    r = stats.linregress(x, y)
    n = len(x)
    tcrit = stats.t.ppf(0.975, n - 2)
    resid = np.asarray(y) - (r.slope * np.asarray(x) + r.intercept)
    rse = float(np.sqrt((resid ** 2).sum() / (n - 2)))
    return r, tcrit, rse


def main():
    model = load_model()
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*mM_LaserRemoved")),
                    key=lambda p: int(re.search(r"(\d+)mM", os.path.basename(p)).group(1))):
        conc = int(re.search(r"(\d+)mM", os.path.basename(d)).group(1))
        for f in sorted(glob.glob(os.path.join(d, "*.asc"))):
            arr = np.loadtxt(f)
            for i in range(1, arr.shape[1]):
                y = arr[:, i]
                if y.max() <= 0:
                    continue
                rows.append((f"{conc}mM_s{i}", conc, i, i_deep(model, y), PAPER[conc]))

    concs = sorted({r[1] for r in rows})
    mu = np.array([np.mean([r[3] for r in rows if r[1] == c]) for c in concs])
    sd = np.array([np.std([r[3] for r in rows if r[1] == c], ddof=1) for c in concs])
    pv = np.array([PAPER[c] for c in concs])

    fm, tm, rsem = fit_stats(concs, mu)
    fp, tp, rsep = fit_stats(concs, pv)

    L = []
    L.append("Deep-layer photon count (I_deep) for the guanine concentration series")
    L.append("=" * 104)
    L.append("")
    L.append("I_deep = (X_max - X_min)/10 * sum of the 216-point deep auxiliary head,")
    L.append("computed with best_raman2_model.pt on Spectrum/guanine_concentrations/*_LaserRemoved.")
    L.append("")
    L.append("'paper' is the value Figure 10(b) of the manuscript reports at the same")
    L.append("concentration (conc_vs_photons in paper_validate_guanine.py).")
    L.append("")
    L.append("The published series and these spectra are different acquisitions, so the two")
    L.append("columns are not expected to agree in absolute terms; the informative comparison")
    L.append("is the shape of the response, given at the end.")
    L.append("")
    L.append("-" * 104)
    L.append(f"{'file':16}{'conc_mM':>10}{'spec':>6}{'I_deep_measured':>20}"
             f"{'I_deep_paper':>18}{'measured/paper':>18}")
    L.append("-" * 104)
    for n, c, i, v, p in rows:
        L.append(f"{n:16}{c:10d}{i:6d}{v:20.1f}{p:18.1f}{v / p:18.2f}")
    L.append("-" * 104)
    L.append("")

    L.append("Per concentration, mean over the five spectra")
    L.append("-" * 104)
    L.append(f"{'conc_mM':>10}{'measured_mean':>18}{'measured_sd':>16}{'RSD_%':>9}"
             f"{'paper':>16}{'measured/paper':>18}")
    L.append("-" * 104)
    for c, m_, s_, p_ in zip(concs, mu, sd, pv):
        L.append(f"{c:10d}{m_:18.1f}{s_:16.1f}{100 * s_ / m_:9.2f}{p_:16.1f}{m_ / p_:18.2f}")
    L.append("-" * 104)
    L.append("")

    L.append("Linear response, measured against published")
    L.append("-" * 104)
    L.append(f"{'':22}{'slope (per mM)':>20}{'intercept':>16}{'R^2':>10}{'residual SE':>16}")
    L.append(f"{'measured':22}{fm.slope:20.1f}{fm.intercept:16.1f}"
             f"{fm.rvalue**2:10.4f}{rsem:16.1f}")
    L.append(f"{'paper, Figure 10(b)':22}{fp.slope:20.1f}{fp.intercept:16.1f}"
             f"{fp.rvalue**2:10.4f}{rsep:16.1f}")
    L.append("")
    L.append(f"measured slope 95% CI: {fm.slope:.1f} +/- {tm * fm.stderr:.1f}")
    L.append(f"measured intercept 95% CI: {fm.intercept:.1f} +/- {tm * fm.intercept_stderr:.1f}")
    L.append(f"slope ratio measured/paper: {fm.slope / fp.slope:.3f}")
    L.append("-" * 104)
    L.append("")

    L.append("Shape of the response, each series normalised to its own 80 mM value")
    L.append("-" * 104)
    L.append("This removes the overall scale difference between the two acquisitions and")
    L.append("compares only how the readout grows with concentration.")
    L.append("")
    L.append(f"{'conc_mM':>10}{'measured_norm':>18}{'paper_norm':>16}{'difference_%':>16}")
    L.append("-" * 104)
    mn, pn = mu / mu[-1], pv / pv[-1]
    for c, a, b in zip(concs, mn, pn):
        L.append(f"{c:10d}{a:18.4f}{b:16.4f}{100 * (a - b) / b:16.1f}")
    L.append("-" * 104)
    L.append(f"mean absolute difference in normalised response: "
             f"{100 * np.abs(mn - pn).mean() / 1:.1f} percentage points of the 80 mM value")
    L.append(f"mean relative difference: {100 * np.abs((mn - pn) / pn).mean():.1f}%")
    L.append("-" * 104)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
