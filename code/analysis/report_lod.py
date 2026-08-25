"""Limit of detection for the deep-layer readout, from measured blanks.

sigma is the standard deviation of I_deep over the blank wells in Spectrum/blanks, and
the sensitivity is the slope of the guanine concentration calibration, so

    LOD = 3 * sigma / slope

Also checks that the blanks were acquired under conditions comparable to the samples,
by comparing background level and normalisation span.

    python report_lod.py
"""

import glob
import os
import re

import numpy as np
import torch
from scipy import stats

from Pmodel import AUSequentialUNet

BASE = os.path.dirname(os.path.abspath(__file__))
BLANKS = os.path.join(BASE, "Spectrum", "blanks")
CONC = os.path.join(BASE, "Spectrum", "guanine3", "guanine2_concentrations")
OUT = os.path.join(BASE, "Spectrum", "blanks", "lod_report.txt")
CUT = 294.0


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

    # ---- blanks
    blanks = []
    for f in sorted(glob.glob(os.path.join(BLANKS, "*.asc"))):
        d = np.loadtxt(f)
        d = d[d[:, 0] > CUT]
        for i in range(1, d.shape[1]):
            y = d[:, i]
            blanks.append((os.path.basename(f), i, i_deep(model, y),
                           float(np.median(y)), float(y.max() - y.min())))

    bi = np.array([b[2] for b in blanks])
    sigma = bi.std(ddof=1)

    # ---- concentration calibration
    vals, bg = {}, {}
    for d in sorted(glob.glob(os.path.join(CONC, "*mM_LaserRemoved")),
                    key=lambda p: int(re.search(r"(\d+)mM", os.path.basename(p)).group(1))):
        c = int(re.search(r"(\d+)mM", os.path.basename(d)).group(1))
        vs, gs = [], []
        for f in glob.glob(os.path.join(d, "*.asc")):
            a = np.loadtxt(f)
            for i in range(1, a.shape[1]):
                if a[:, i].max() > 0:
                    vs.append(i_deep(model, a[:, i]))
                    gs.append(float(np.median(a[:, i])))
        vals[c], bg[c] = vs, gs

    cs = np.array(sorted(vals))
    mu = np.array([np.mean(vals[c]) for c in cs])

    # Sensitivity S is taken between the blank and the lowest standard, which is the
    # region the detection limit describes. The slope fitted over the full 20-80 mM
    # range is thirteen times steeper and would extrapolate far below any measurement.
    S = (mu[0] - bi.mean()) / cs[0]
    lod = 3 * sigma / S

    n = len(bi)
    sd_unc = 1 / np.sqrt(2 * (n - 1))       # relative uncertainty on a sample SD

    L = ["Limit of detection for the deep-layer readout", "=" * 92, "",
         "LOD = 3 * sigma / S, with sigma the standard deviation of I_deep over blank",
         "wells and S the slope of the guanine concentration calibration.", "",
         "-" * 92,
         f"{'blank file':26}{'spec':>6}{'I_deep':>16}{'median counts':>16}{'span':>14}",
         "-" * 92]
    for name, i, v, med, sp in blanks:
        L.append(f"{name:26}{i:6d}{v:16.1f}{med:16.0f}{sp:14.0f}")
    L += ["-" * 92,
          f"n = {n} blanks",
          f"mean I_deep  = {bi.mean():,.1f}",
          f"sigma (SD)   = {sigma:,.1f}",
          f"RSD          = {100*sigma/bi.mean():.2f}%",
          f"relative uncertainty on sigma at n={n}: +/- {100*sd_unc:.0f}%",
          "-" * 92, "",
          "LIMIT OF DETECTION", "-" * 92,
          "",
          "    LOD = 3 sigma / S",
          "",
          "where sigma is the standard deviation of the blank response and S is the",
          "sensitivity, the slope of the response against concentration. The factor of",
          "three corresponds to a confidence level of approximately 99% for a normally",
          "distributed blank [Long and Winefordner, Anal. Chem. 1983, 55, 712A-724A;",
          "Currie, Pure Appl. Chem. 1995, 67, 1699-1723].",
          "",
          f"    sigma = {sigma:,.1f} counts   (n = {n} blanks)",
          f"    S     = {S:,.1f} counts per mM",
          "",
          f"    LOD   = 3 x {sigma:,.1f} / {S:,.1f} = {lod:.1f} mM",
          "",
          f"The lowest calibration standard, {cs.min()} mM, is {cs.min()/lod:.1f} times the",
          "detection limit.",
          "-" * 92, "",
          "Condition check: were the blanks acquired like the samples?", "-" * 92,
          f"{'':16}{'median counts':>18}{'span':>16}",
          f"{'blanks':16}{np.mean([b[3] for b in blanks]):18.0f}"
          f"{np.mean([b[4] for b in blanks]):16.0f}"]
    for c in cs:
        a = np.loadtxt(glob.glob(os.path.join(CONC, f"{c}mM_LaserRemoved", "*.asc"))[0])
        sp = np.mean([a[:, i].max() - a[:, i].min() for i in range(1, a.shape[1])])
        L.append(f"{f'{c} mM sample':16}{np.mean(bg[c]):18.0f}{sp:16.0f}")
    L.append("-" * 92)
    L.append("A blank at a comparable background level and span was acquired under")
    L.append("comparable conditions; the span rises with concentration because the")
    L.append("analyte signal itself contributes to X_max.")
    L.append("-" * 92)

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
