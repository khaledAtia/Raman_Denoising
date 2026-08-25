"""Replace the Raman-shift column of the laser-removed NewData spectra with the
axis of the older acquisitions the model was trained and validated on.

Both sets are 864 points after laser removal, so the substitution is a straight
column swap: the intensities are untouched and only the wavenumber label of each
sample index changes. Whether that relabelling is *correct* is an empirical
question, which check_axis_alignment.py answers by comparing recovered band
positions against the values published for guanine.

    python substitute_axis.py                 # Guanine
    python substitute_axis.py Tyrosin
"""

import glob
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(BASE, "Spectrum", "Glycerin2", "200_P.txt")
N_POINTS = 864


def reference_axis():
    axes = []
    for f in sorted(glob.glob(os.path.join(BASE, "Spectrum", "Glycerin2", "*_P.txt"))):
        axes.append(np.loadtxt(f, skiprows=1)[:, 0])
    for a in axes[1:]:
        if not np.allclose(a, axes[0], atol=1e-6):
            sys.exit("the reference files do not share one axis; cannot substitute")
    a = axes[0]
    if a.size != N_POINTS:
        sys.exit(f"reference axis is {a.size} points, expected {N_POINTS}")
    print(f"reference axis from Spectrum/Glycerin2 ({len(axes)} files agree): "
          f"{a[0]:.3f} .. {a[-1]:.3f} cm-1, {a.size} points")
    return a


def process(folder, axis):
    src = os.path.join(BASE, "NewData", f"{folder}_LaserRemoved")
    dst = os.path.join(BASE, "NewData", f"{folder}_LaserRemoved_OldAxis")
    if not os.path.isdir(src):
        sys.exit(f"run prepare_newdata.py first; missing {src}")
    os.makedirs(dst, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "*.asc")))
    n = 0
    for f in files:
        d = np.loadtxt(f)
        if d.shape[0] != N_POINTS:
            print(f"  SKIP {os.path.basename(f)}: {d.shape[0]} points")
            continue
        d[:, 0] = axis
        np.savetxt(os.path.join(dst, os.path.basename(f)), d,
                   fmt="%.6f" + "\t%.0f" * (d.shape[1] - 1), delimiter="\t")
        n += 1
    print(f"{folder}: wrote {n} files -> {os.path.relpath(dst, BASE)}")


if __name__ == "__main__":
    ax = reference_axis()
    for folder in (sys.argv[1:] or ["Guanine"]):
        process(folder, ax)
