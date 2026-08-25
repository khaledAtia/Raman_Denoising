"""Calibrate and laser-trim the NewData acquisitions to the 864-point network input.

The raw .asc files are 1024 x 6: column 0 is a Raman shift, columns 1-5 are five
independent spectra of the same well. The wavenumber column shipped inside those files
is NOT the instrument's calibrated axis. The calibrated axis is supplied separately in
``NewData/data_calibrated.asc`` (1024 points, -96.33 to 1888.19 cm-1), and it is the
same calibration under which the data reported in the paper were acquired: its points
160..1023 are identical, to machine precision, to the 864-point axis of
``Spectrum/Glycerin2``.

The preprocessing is therefore

  1. replace column 0 with the calibrated axis, sample for sample;
  2. apply the project's laser cut, ``remove_laser(data, 294)``.

Step 2 keeps exactly points 160..1023, so the result is 864 points on precisely the
axis the network was trained on -- no interpolation and no arbitrary trimming.

    python prepare_newdata.py                 # processes Guanine
    python prepare_newdata.py Tyrosin         # or any other NewData subfolder
"""

import glob
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CALIB = os.path.join(BASE, "NewData", "data_calibrated.asc")
TRAIN_AXIS = os.path.join(BASE, "Spectrum", "Glycerin2", "200_P.txt")
CUT = 294.0          # cm-1, the project's laser-removal threshold
N_POINTS = 864       # the network's input length


def remove_laser(data, spectrum_cut=CUT):
    """Identical to RamanUtils.RamanHelpers.remove_laser, inlined so this script
    does not depend on the environment in which that package is installed."""
    return data[data[:, 0] > spectrum_cut, :]


def calibrated_axis():
    axis = np.loadtxt(CALIB)[:, 0]
    kept = axis[axis > CUT]
    if kept.size != N_POINTS:
        sys.exit(f"calibrated axis gives {kept.size} points above {CUT}, expected {N_POINTS}")
    train = np.loadtxt(TRAIN_AXIS, skiprows=1)[:, 0]
    dev = np.abs(kept - train).max()
    print(f"calibrated axis: {axis.size} points, {axis[0]:.2f} .. {axis[-1]:.2f} cm-1")
    print(f"  after the {CUT:.0f} cm-1 cut: {kept.size} points, "
          f"{kept[0]:.3f} .. {kept[-1]:.3f} cm-1")
    print(f"  agreement with the training axis: max deviation {dev:.2e} cm-1")
    return axis


def process(folder, axis):
    src = os.path.join(BASE, "NewData", folder)
    dst = os.path.join(BASE, "NewData", f"{folder}_Calibrated")
    if not os.path.isdir(src):
        sys.exit(f"no such folder: {src}")
    os.makedirs(dst, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "*.asc")))
    if not files:
        sys.exit(f"no .asc files in {src}")

    print(f"\n{folder}: {len(files)} files -> {os.path.relpath(dst, BASE)}")
    skipped, flagged = [], []

    for f in files:
        name = os.path.basename(f)
        raw = np.loadtxt(f)
        if raw.shape[0] != axis.size:
            skipped.append((name, raw.shape[0]))
            continue

        raw[:, 0] = axis                      # calibrate
        out = remove_laser(raw)               # then trim the Rayleigh wing

        if out.shape[0] != N_POINTS:
            skipped.append((name, out.shape[0]))
            continue
        if not np.isfinite(out).all():
            flagged.append(f"{name}: non-finite values")

        dead = [i for i in range(1, out.shape[1]) if out[:, i].max() <= 0]
        if dead:
            flagged.append(f"{name}: spectra {dead} are all zero")
        hot = [i for i in range(1, out.shape[1]) if (out[:, i] >= 65535).any()]
        if hot:
            flagged.append(f"{name}: spectra {hot} contain saturated points")

        np.savetxt(os.path.join(dst, name), out,
                   fmt="%.6f" + "\t%.0f" * (out.shape[1] - 1), delimiter="\t")

    written = sorted(glob.glob(os.path.join(dst, "*.asc")))
    print(f"  wrote {len(written)} files")
    if written:
        a = np.loadtxt(written[0])
        print(f"  {a.shape[0]} points, {a.shape[1] - 1} spectra per file, "
              f"{a[0, 0]:.3f} .. {a[-1, 0]:.3f} cm-1")
    if skipped:
        print(f"  SKIPPED: {skipped}")
    for msg in flagged:
        print(f"  NOTE {msg}")


if __name__ == "__main__":
    ax = calibrated_axis()
    for folder in (sys.argv[1:] or ["Guanine"]):
        process(folder, ax)
