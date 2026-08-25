"""Is the substituted axis correct?

The manuscript reports guanine bands, measured on the older acquisitions, at
654, 1165, 1200, 1230, 1270, 1335, 1380, 1460 and 1540 cm-1. Those positions are the
yardstick. We baseline-correct the new guanine spectra, locate their peaks by sample
index, and then read those indices off under two candidate calibrations:

  own  - the axis shipped with the new .asc files
  old  - the axis of the older acquisitions, substituted in by substitute_axis.py

Whichever reproduces the published positions is the correct labelling for this data.

    python check_axis_alignment.py
"""

import glob
import os

import numpy as np
from scipy.signal import find_peaks

from RamanUtils.airPLS import airPLS

BASE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = [654, 1165, 1200, 1230, 1270, 1335, 1380, 1460, 1540]


def corrected(path):
    """Mean of the valid spectra in a file, baseline-corrected."""
    d = np.loadtxt(path)
    cols = [d[:, i] for i in range(1, d.shape[1]) if d[:, i].max() > 0]
    y = np.mean(cols, axis=0)
    return d[:, 0], y - airPLS(y, 1e4, 2, 40)


def peak_indices(y, n=25):
    p, props = find_peaks(y, prominence=(y.max() - y.min()) * 0.02, distance=4)
    order = np.argsort(props["prominences"])[::-1]
    return sorted(p[order[:n]])


def score(peaks_cm):
    """Nearest detected peak to each published band, and the spread of the misses."""
    dev = [min(peaks_cm, key=lambda p: abs(p - r)) - r for r in REFERENCE]
    return dev, float(np.mean(np.abs(dev))), float(np.sqrt(np.mean(np.square(dev))))


def main():
    own_dir = os.path.join(BASE, "NewData", "Guanine_Calibrated")
    old_dir = os.path.join(BASE, "NewData", "Guanine_LaserRemoved")

    # the strongest acquisitions give the most reliable peak positions
    names = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(own_dir, "*.asc")))
             if "60s" in f or "50s" in f]
    print(f"using {len(names)} long-exposure files\n")

    rows = []
    for name in names:
        ax_own, y = corrected(os.path.join(own_dir, name))
        ax_old = np.loadtxt(os.path.join(old_dir, name))[:, 0]
        idx = peak_indices(y)
        d_own, m_own, r_own = score(ax_own[idx])
        d_old, m_old, r_old = score(ax_old[idx])
        rows.append((name, m_own, r_own, m_old, r_old))

    print(f"{'file':30}{'calibrated axis':>20}{'file-supplied axis':>20}")
    print(f"{'':30}{'mean|dev|':>10}{'rms':>10}{'mean|dev|':>10}{'rms':>10}")
    for name, mo, ro, ml, rl in rows:
        print(f"{name[:29]:30}{mo:10.1f}{ro:10.1f}{ml:10.1f}{rl:10.1f}")
    a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print(f"{'MEAN OVER FILES':30}{a[:,0].mean():10.1f}{a[:,1].mean():10.1f}"
          f"{a[:,2].mean():10.1f}{a[:,3].mean():10.1f}")

    # band-by-band detail on the single strongest acquisition
    name = "guanine 80 mM 60s.asc"
    ax_own, y = corrected(os.path.join(own_dir, name))
    ax_old = np.loadtxt(os.path.join(old_dir, name))[:, 0]
    idx = peak_indices(y)
    d_own, _, _ = score(ax_own[idx])
    d_old, _, _ = score(ax_old[idx])
    print(f"\nband by band, {name}:")
    print(f"{'published':>10}{'calibrated axis':>22}{'file-supplied':>22}")
    for r, do, dl in zip(REFERENCE, d_own, d_old):
        print(f"{r:>10}{r+do:>12.1f} ({do:+6.1f}){r+dl:>12.1f} ({dl:+6.1f})")


if __name__ == "__main__":
    main()
