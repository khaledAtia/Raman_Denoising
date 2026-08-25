"""Train the fully convolutional encoder-decoder denoiser on our synthetic engine.

Third structure in the comparison of SI Section S5, alongside the dual-branch
network and the cascaded U-Net. Everything outside the model is shared with
train_ablation.py and train_kazemzadeh.py: the same generator, the same
signal-to-noise ranges, the same frozen validation set, the same seed, the same
optimiser schedule and early stopping, and the same
ContrastiveLoss.compute_signal_loss for scoring, so Val MAE / cosine / shape are
directly comparable with the other two.

    python train_fcn_encdec.py --seed 0
"""

import argparse
import json
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import train_ablation as T
from Pmodel import ContrastiveLoss, EarlyStopping
from fcn_encdec_model import FCNEncoderDecoder

LEARNING_RATE = T.LEARNING_RATE
WEIGHT_DECAY = 0.0


def signal_metrics(criterion, pred, target):
    """Returns the differentiable loss plus the three metrics as plain floats."""
    loss, mae, cos, shape = criterion.compute_signal_loss(pred, target, apply_shape=True)
    f = lambda v: float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
    return loss, f(mae), f(cos), f(shape)


def train(args):
    tag = args.tag or f"fcnencdec_seed{args.seed}"
    os.makedirs(T.RUNS_DIR, exist_ok=True)
    os.makedirs(T.MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(T.RUNS_DIR, f"{tag}_best.pt")
    hist_csv = os.path.join(T.RUNS_DIR, f"{tag}_history.csv")
    meta_path = os.path.join(T.RUNS_DIR, f"{tag}_meta.json")
    plot_path = os.path.join(T.RUNS_DIR, f"{tag}_loss.png")
    model_path = os.path.join(T.MODELS_DIR,
                              f"Compare_fcn_encoder_decoder_seed{args.seed}.pth")

    print("=" * 78)
    print(f"  FULLY CONVOLUTIONAL ENCODER-DECODER (after Loc et al. 2022)")
    print(f"  seed={args.seed}   tag={tag}   device: {T.DEVICE}")
    print("=" * 78)

    # --- data: identical stream and val set to the other runs ---------------
    T.set_seed(args.seed)
    train_gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                                     epoch_size=T.TRAIN_EPOCH_SIZE,
                                     min_snr=T.TRAIN_MIN_SNR, max_snr=T.TRAIN_MAX_SNR)
    train_dl = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True)

    val_gen = T.RamanDataGenerator(T.BASELINE_FILE, T.NOISE_FILE,
                                   min_snr=T.VAL_MIN_SNR, max_snr=T.VAL_MAX_SNR,
                                   blank_ratio=T.VAL_BLANK_RATIO)
    val_ds = T.FixedRamanDataset(val_gen, n_samples=T.VAL_N_SAMPLES)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"train {len(train_gen)}/epoch (SNR {T.TRAIN_MIN_SNR}-{T.TRAIN_MAX_SNR}) | "
          f"fixed val {len(val_ds)} (SNR {T.VAL_MIN_SNR}-{T.VAL_MAX_SNR})")

    # --- model / optimiser / loss -------------------------------------------
    model = FCNEncoderDecoder().to(T.DEVICE)
    n_total, _ = T.count_parameters(model)
    print(f"Parameters: {n_total:,}  (cf. 5,168,630 dual-branch, 5,888,130 cascade)")

    latency = T.benchmark_inference(model, train_gen.n_points, T.DEVICE)
    print(f"Latency: {latency['batch1_ms']:.3f} ms/spectrum (b1), "
          f"{latency['plate96_ms']:.3f} ms/96-well plate")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=T.SCHED_PATIENCE,
        factor=T.SCHED_FACTOR, min_lr=T.SCHED_MIN_LR)
    early = EarlyStopping(patience=T.ES_PATIENCE, start_epoch=T.ES_START_EPOCH,
                          delta=T.ES_DELTA, verbose=True, path=ckpt_path)
    criterion = ContrastiveLoss()

    hist = {"train_loss": [], "val_loss": [], "val_mae": [], "val_cos": [], "val_shape": []}
    epoch_times = []
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        te = time.perf_counter()
        model.train()
        run = 0.0
        for x, y_s, y_b in train_dl:
            x, y_s = x.to(T.DEVICE), y_s.to(T.DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss, _, _, _ = signal_metrics(criterion, pred, y_s)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=T.GRAD_CLIP_NORM)
            optimizer.step()
            run += loss.item()
        hist["train_loss"].append(run / len(train_dl))

        model.eval()
        vl = vmae = vcos = vsh = 0.0
        with torch.no_grad():
            for x, y_s, y_b in val_dl:
                x, y_s = x.to(T.DEVICE), y_s.to(T.DEVICE)
                pred = model(x)
                lf, mae, cos, sh = signal_metrics(criterion, pred, y_s)
                vl += lf.item(); vmae += mae; vcos += cos; vsh += sh
        nb = len(val_dl)
        vmae /= nb; vcos /= nb; vsh /= nb
        hist["val_loss"].append(vl / nb); hist["val_mae"].append(vmae)
        hist["val_cos"].append(vcos); hist["val_shape"].append(vsh)

        scheduler.step(vmae)
        epoch_times.append(time.perf_counter() - te)
        print(f"[fcn] Epoch {epoch+1}/{args.epochs} | "
              f"LR {optimizer.param_groups[0]['lr']*1e4:.4f} | Val MAE {vmae:.4f} | "
              f"Val Cos {vcos:.4f} | Val Shape {vsh:.4f} | {epoch_times[-1]:.1f}s")

        early(vmae, model, epoch)
        if early.early_stop:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    total_time = time.perf_counter() - t0
    print(f"Done in {total_time/3600:.2f} h ({np.mean(epoch_times):.1f} s/epoch).")

    # --- restore best, re-score, save ---------------------------------------
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print("Loaded best weights.")
    else:
        print(f"WARNING: no checkpoint (ended before start_epoch={T.ES_START_EPOCH}).")

    model.eval()
    r_mae = r_cos = r_sh = 0.0
    with torch.no_grad():
        for x, y_s, y_b in val_dl:
            x, y_s = x.to(T.DEVICE), y_s.to(T.DEVICE)
            pred = model(x)
            _, mae, cos, sh = signal_metrics(criterion, pred, y_s)
            r_mae += mae; r_cos += cos; r_sh += sh
    nb = len(val_dl)
    restored = {"val_mae": r_mae/nb, "val_cos": r_cos/nb, "val_shape": r_sh/nb}
    print(f"Restored-checkpoint metrics | Val MAE {restored['val_mae']:.6f} | "
          f"Val Cos {restored['val_cos']:.6f} | Val Shape {restored['val_shape']:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    ep = range(1, len(hist["train_loss"]) + 1)
    np.savetxt(hist_csv,
               np.column_stack((list(ep), hist["train_loss"], hist["val_loss"],
                                hist["val_mae"], hist["val_cos"], hist["val_shape"],
                                epoch_times)),
               delimiter=",", header="epoch,train_loss,val_loss,val_mae,val_cos,"
                                     "val_shape,epoch_seconds", comments="")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    ax[0].plot(list(ep), hist["train_loss"], label="train")
    ax[0].plot(list(ep), hist["val_loss"], label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("signal loss"); ax[0].legend(frameon=False)
    ax[1].plot(list(ep), hist["val_mae"], label="val MAE")
    ax[1].plot(list(ep), hist["val_cos"], label="val cosine")
    ax[1].set_xlabel("epoch"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(plot_path); plt.close(fig)

    meta = {
        "tag": tag,
        "model": "FCNEncoderDecoder (after Loc et al. 2022)",
        "module": "fcn_encdec_model",
        "note": "architecture family, not a reproduction of published hyperparameters",
        "seed": args.seed,
        "device": str(T.DEVICE),
        "params_total": n_total,
        "latency": latency,
        "epochs_run": len(hist["train_loss"]),
        "mean_epoch_seconds": round(float(np.mean(epoch_times)), 3),
        "total_train_hours": round(total_time / 3600, 4),
        "restored_val_mae": round(restored["val_mae"], 6),
        "restored_val_cos": round(restored["val_cos"], 6),
        "restored_val_shape": round(restored["val_shape"], 6),
        "argmin_epoch": int(np.argmin(hist["val_mae"])) + 1,
        "argmin_val_mae": round(float(np.min(hist["val_mae"])), 6),
        "model_path": model_path,
        "config": {"learning_rate": LEARNING_RATE, "batch_size": args.batch_size,
                   "train_snr": [T.TRAIN_MIN_SNR, T.TRAIN_MAX_SNR],
                   "val_snr": [T.VAL_MIN_SNR, T.VAL_MAX_SNR],
                   "val_n_samples": T.VAL_N_SAMPLES,
                   "train_epoch_size": T.TRAIN_EPOCH_SIZE},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=T.BATCH_SIZE)
    p.add_argument("--tag", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
