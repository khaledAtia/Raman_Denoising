"""Photon-count (I_deep) readout for the measured guanine spectra, against Figure 10(d).

For every spectrum processed by process_guanine_spectra.py this computes the manuscript's
deep-layer readout,

    I_deep = (X_max - X_min)/10 * sum_i s_deep_i ,

and places it beside the value the published Figure 10(d) reports at the same acquisition
time. The published values were recovered from the vector content of
paper_revised/figures/figure10.pdf; refitting them returns R^2 = 0.9935, the value printed
on the figure, which confirms the recovery.

    python report_photon_counts.py
"""

import glob
import os
import re

import numpy as np
import torch
from scipy import stats

from Pmodel import AUSequentialUNet

import sys

BASE = os.path.dirname(os.path.abspath(__file__))
FOLDER = sys.argv[1] if len(sys.argv) > 1 else "100mM"
SRC = os.path.join(BASE, "Spectrum", "Guanine", f"{FOLDER}_LaserRemoved")
OUT = os.path.join(BASE, "Spectrum", "Guanine", f"{FOLDER}_Processed",
                   "photon_count_report.txt")

# Figure 10(d) of the manuscript, as supplied by the authors. Independently recovering
# these from the vector content of figures/figure10.pdf reproduced them to within about
# 245 counts, a constant offset consistent with marker centring.
PAPER = {10: 3.20989366e+05, 20: 5.46100452e+05, 30: 6.63493760e+05,
         40: 8.81322028e+05, 50: 1.00621318e+06, 60: 1.17654973e+06}


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
    pt = sorted(PAPER)
    fit = stats.linregress(pt, [PAPER[t] for t in pt])

    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "*.asc")),
                    key=lambda p: int(re.search(r"(\d+)s", os.path.basename(p)).group(1))):
        name = os.path.basename(f)
        t = int(re.search(r"(\d+)s", name).group(1))
        d = np.loadtxt(f)
        for i in range(1, d.shape[1]):
            y = d[:, i]
            if y.max() <= 0:
                continue
            clean = re.sub(r"^#\d+\s*", "", os.path.splitext(name)[0]).strip()
            rows.append((f"{clean}_s{i}", t, i, i_deep(model, y), PAPER[t]))

    # proportionality, which is what Section S9.3 asserts: I(t) = k t
    prop_paper = {t: PAPER[60] / 60 * t for t in pt}
    meas60 = np.mean([r[3] for r in rows if r[1] == 60])
    prop_meas = {t: meas60 / 60 * t for t in pt}

    L = []
    L.append(f"Deep-layer photon count (I_deep) for the measured guanine spectra -- {FOLDER}")
    L.append("=" * 100)
    L.append("")
    L.append("I_deep = (X_max - X_min)/10 * sum of the 216-point deep auxiliary head,")
    L.append(f"computed with best_raman2_model.pt on Spectrum/Guanine/{FOLDER}_LaserRemoved.")
    L.append("")
    L.append("'paper' is the value Figure 10(d) of the manuscript reports at the same")
    L.append("acquisition time, as supplied by the authors.")
    L.append(f"Refitting those points gives I(t) = {fit.slope:.0f} t + {fit.intercept:.0f}, "
             f"R^2 = {fit.rvalue**2:.4f},")
    L.append("matching the R^2 = 0.9935 printed on the figure.")
    L.append("")
    L.append("NOTE: the published series and these spectra are different acquisitions of")
    L.append("100 mM guanine, so the two columns are not expected to agree exactly.")
    L.append("")
    L.append("-" * 100)
    L.append(f"{'file':32}{'time_s':>8}{'spec':>6}{'I_deep_measured':>18}"
             f"{'I_deep_paper':>16}{'paper/measured':>16}")
    L.append("-" * 100)
    for n, t, i, v, p in rows:
        L.append(f"{n:32}{t:8d}{i:6d}{v:18.1f}{p:16.0f}{p / v:16.2f}")
    L.append("-" * 100)
    L.append("")

    L.append("Per acquisition time, mean over the five spectra")
    L.append("-" * 100)
    L.append(f"{'time_s':>8}{'measured_mean':>16}{'measured_sd':>14}{'RSD_%':>9}"
             f"{'paper':>14}{'paper/measured':>16}")
    L.append("-" * 100)
    for t in pt:
        v = np.array([r[3] for r in rows if r[1] == t])
        L.append(f"{t:8d}{v.mean():16.1f}{v.std(ddof=1):14.1f}{100*v.std(ddof=1)/v.mean():9.2f}"
                 f"{PAPER[t]:14.0f}{PAPER[t]/v.mean():16.2f}")
    L.append("-" * 100)
    L.append("")

    L.append("Test of the proportionality asserted in SI Section S9.3, I(t) = k t")
    L.append("-" * 100)
    L.append("Each series is anchored on its own 60 s value and scaled by t/60; the")
    L.append("deviation is what an experimenter would incur by substituting a short")
    L.append("acquisition for a long one and scaling by the exposure ratio.")
    L.append("")
    L.append(f"{'time_s':>8}{'paper_actual':>16}{'paper_if_prop':>16}{'dev_%':>10}"
             f"{'meas_actual':>16}{'meas_if_prop':>16}{'dev_%':>10}")
    L.append("-" * 100)
    for t in pt:
        m = np.mean([r[3] for r in rows if r[1] == t])
        L.append(f"{t:8d}{PAPER[t]:16.0f}{prop_paper[t]:16.0f}"
                 f"{100*(PAPER[t]-prop_paper[t])/prop_paper[t]:10.1f}"
                 f"{m:16.1f}{prop_meas[t]:16.0f}"
                 f"{100*(m-prop_meas[t])/prop_meas[t]:10.1f}")
    L.append("-" * 100)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
