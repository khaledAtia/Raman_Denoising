"""
Re-evaluate a saved ablation checkpoint on the fixed validation set.

Use this to recover the authoritative metrics for runs completed BEFORE the
restored-checkpoint fix was added to train_ablation.py, without retraining.

The validation set is rebuilt with the same seed and the same generator settings
as the training run, so it is bit-identical to the one used during training.

    python reeval_checkpoint.py --arch rk4 --seed 0

Prints the metrics of the weights on disk and, with --update-meta, writes them
into the run's meta.json under the same keys train_ablation.py now uses.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import train_ablation as T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=sorted(T.ARCHITECTURES), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--checkpoint", default=None,
                   help="defaults to runs/<tag>_best.pt")
    p.add_argument("--update-meta", action="store_true",
                   help="write the recovered metrics into runs/<tag>_meta.json")
    args = p.parse_args()

    tag = args.tag or f"{args.arch}_seed{args.seed}"
    ckpt = args.checkpoint or os.path.join(T.RUNS_DIR, f"{tag}_best.pt")
    meta_path = os.path.join(T.RUNS_DIR, f"{tag}_meta.json")

    if not os.path.isfile(ckpt):
        raise SystemExit(f"checkpoint not found: {ckpt}")

    # Rebuild the identical validation set: seed first, then construct the
    # training generator (which consumes RNG draws) before the validation one,
    # exactly as train_ablation.py does.
    T.set_seed(args.seed)
    _ = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                             epoch_size=T.TRAIN_EPOCH_SIZE,
                             min_snr=T.TRAIN_MIN_SNR, max_snr=T.TRAIN_MAX_SNR)
    val_gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                                   min_snr=T.VAL_MIN_SNR, max_snr=T.VAL_MAX_SNR,
                                   blank_ratio=T.VAL_BLANK_RATIO)
    val_ds = T.FixedRamanDataset(val_gen, n_samples=T.VAL_N_SAMPLES)
    val_dl = DataLoader(val_ds, batch_size=T.BATCH_SIZE, shuffle=False)

    model = T.ARCHITECTURES[args.arch](
        n_channels=1, bilinear=False, gamma=T.GAMMA, kernels=T.KERNELS,
        base_latent_dim=T.BASE_LATENT_DIM, use_derivatives=T.USE_DERIVATIVES).to(T.DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE, weights_only=True))
    model.eval()

    criterion = T.ContrastiveLoss(
        w_signal=T.W_SIGNAL, w_base=T.W_BASE, w_consist=T.W_CONSIST,
        w_smooth=T.W_SMOOTH, w_curve=T.W_CURVE, dip_factor=T.D_FACTOR,
        w_mae=T.W_MAE, w_cos=T.W_COS, w_shape=T.W_SHAPE,
        w_mid=T.W_MID, w_deep=T.W_DEEP, w_ortho=T.W_ORTHO)

    mae = cos = shape = 0.0
    with torch.no_grad():
        for inputs, target_s, target_b in val_dl:
            inputs   = inputs.to(T.DEVICE)
            target_s = target_s.to(T.DEVICE)
            target_b = target_b.to(T.DEVICE)
            pred_s, pred_b, _, _, _, _, x4b, x4s = model(inputs)
            _, comp = criterion(pred_s, target_s, pred_b, target_b, inputs,
                                x4_base=x4b, x4_sig=x4s)
            mae   += comp[0]
            cos   += comp[1]
            shape += comp[2]
    n = len(val_dl)
    out = {"restored_val_mae": round(mae / n, 6),
           "restored_val_cos": round(cos / n, 6),
           "restored_val_shape": round(shape / n, 6)}

    print(f"\ncheckpoint : {ckpt}")
    print(f"arch/seed  : {args.arch} / {args.seed}")
    for k, v in out.items():
        print(f"{k:20s}: {v}")

    if args.update_meta:
        if not os.path.isfile(meta_path):
            raise SystemExit(f"meta.json not found: {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        # Preserve the old argmin-based fields under their new names.
        for old, new in (("best_epoch", "argmin_epoch"),
                         ("best_val_mae", "argmin_val_mae"),
                         ("val_cos_at_best", "argmin_val_cos"),
                         ("val_shape_at_best", "argmin_val_shape")):
            if old in meta:
                meta[new] = meta.pop(old)
        meta.update(out)
        meta["reeval_note"] = ("restored_* metrics recovered post hoc by "
                               "reeval_checkpoint.py; argmin_* are the raw "
                               "training-history minima and describe weights "
                               "that were not saved")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nupdated {meta_path}")


if __name__ == "__main__":
    main()
