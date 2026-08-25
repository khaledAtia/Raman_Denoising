"""Run the trained dual-branch model over the measured guanine spectra.

For every .asc under Spectrum/Guanine/<subdir>/ the script

  1. applies the project's laser cut, remove_laser(data, 294), which leaves the
     864 points the network takes, and writes the model-ready file to
     Spectrum/Guanine/<subdir>_LaserRemoved/;
  2. runs each of the five spectra in the file through the unchanged trained model;
  3. writes one four-column text file per spectrum to
     Spectrum/Guanine/<subdir>_Processed/, in the same layout as
     Spectrum/Glycerin2/*_P.txt:

         Raman_Shift  Original_Spectrum  Predicted_Raman  Predicted_Baseline

The inference path and the rescaling back to detector counts are the ones already
used elsewhere in this repository, verified against Spectrum/Glycerin2/200_P.txt to
within 1.4 counts:

    span = X_max - X_min                       (per spectrum)
    Predicted_Raman    = s_norm * span / 10    (the factor of 10 is the training scale)
    Predicted_Baseline = b_norm * span + X_min

    python process_guanine_spectra.py
"""

import glob
import os
import re
import sys

import numpy as np
import torch

from Pmodel import AUSequentialUNet

BASE = os.path.dirname(os.path.abspath(__file__))
# Root holding one sub-folder per sample; override on the command line, e.g.
#   python process_guanine_spectra.py guanine_concentrations
ROOT = os.path.join(BASE, "Spectrum", sys.argv[1] if len(sys.argv) > 1 else "Guanine")
CUT = 294.0
N_POINTS = 864
HEADER = "Raman_Shift Original_Spectrum Predicted_Raman Predicted_Baseline"


def remove_laser(data, spectrum_cut=CUT):
    """Identical to RamanUtils.RamanHelpers.remove_laser."""
    return data[data[:, 0] > spectrum_cut, :]


def load_model():
    m = AUSequentialUNet(n_channels=1, bilinear=False, gamma=1.0, kernels=[1, 1, 3],
                         base_latent_dim=16, use_derivatives=False)
    m.load_state_dict(torch.load(os.path.join(BASE, "best_raman2_model.pt"),
                                 map_location="cpu", weights_only=True))
    m.eval()
    return m


def infer(model, raw):
    """Recovered Raman signal and baseline, both in detector counts."""
    lo, hi = float(raw.min()), float(raw.max())
    span = hi - lo
    t = torch.tensor((raw - lo) / span, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        o = model(t)
    signal = o[0].squeeze().numpy() * span / 10.0
    baseline = o[1].squeeze().numpy() * span + lo
    return signal, baseline


def stem(name):
    """'#0   Acquisition_10s.asc' -> 'Acquisition_10s'."""
    s = os.path.splitext(name)[0]
    s = re.sub(r"^#\d+\s*", "", s).strip()
    return re.sub(r"\s+", "_", s) or "spectrum"


def process(subdir, model):
    src = os.path.join(ROOT, subdir)
    cut_dir = os.path.join(ROOT, f"{subdir}_LaserRemoved")
    out_dir = os.path.join(ROOT, f"{subdir}_Processed")
    os.makedirs(cut_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "*.asc")) + glob.glob(os.path.join(src, "*.txt")))
    print(f"\n{subdir}: {len(files)} files")
    written, notes = 0, []

    for f in files:
        name = os.path.basename(f)
        raw = np.loadtxt(f)
        cut = remove_laser(raw)
        if cut.shape[0] != N_POINTS:
            notes.append(f"{name}: {cut.shape[0]} points after the cut, expected {N_POINTS} -- skipped")
            continue

        np.savetxt(os.path.join(cut_dir, name), cut,
                   fmt="%.6f" + "\t%.0f" * (cut.shape[1] - 1), delimiter="\t")

        n_spec = cut.shape[1] - 1
        if n_spec != 5:
            notes.append(f"{name}: {n_spec} spectra, expected 5")

        x = cut[:, 0]
        for i in range(1, cut.shape[1]):
            y = cut[:, i]
            if y.max() <= 0:
                notes.append(f"{name}: spectrum {i} is all zero -- not processed")
                continue
            # Real clipping pins several points at one identical maximum. A fixed
            # 65535 threshold is wrong here: some acquisitions legitimately exceed it.
            if (y == y.max()).sum() >= 3:
                notes.append(f"{name}: spectrum {i} appears clipped at {y.max():.0f} "
                             f"(processed anyway; the span, and so the scaling, is affected)")
            sig, base = infer(model, y)
            out = np.column_stack([x, y, sig, base])
            np.savetxt(os.path.join(out_dir, f"{stem(name)}_s{i}_P.txt"), out,
                       fmt="%.6f", delimiter=" ", header=HEADER, comments="")
            written += 1

    print(f"  laser-removed inputs -> {os.path.relpath(cut_dir, BASE)}")
    print(f"  processed spectra    -> {os.path.relpath(out_dir, BASE)}  ({written} files)")
    for n in notes:
        print(f"  NOTE {n}")
    return written


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"no such folder: {ROOT}")
    subs = [d for d in sorted(os.listdir(ROOT))
            if os.path.isdir(os.path.join(ROOT, d))
            and not d.endswith(("_Processed", "_LaserRemoved"))]
    if not subs:
        sys.exit(f"no sample folders under {ROOT}")

    model = load_model()
    total = sum(process(s, model) for s in subs)
    print(f"\ntotal spectra processed: {total}")


if __name__ == "__main__":
    main()
