"""
Train the cascaded-U-Net comparator (Kazemzadeh et al. 2022) on OUR data, for a head-to-head
against the dual-branch model. Answers R1.2 / R2.4 / R3.7.

The point of the exercise is a FAIR comparison, so this script reuses train_ablation.py's
data pipeline verbatim:
  - same seed, so the NumPy-driven training stream is bit-identical to our own runs;
  - the same frozen 2000-spectrum validation set, rebuilt exactly as reeval_checkpoint.py
    does (seed, throwaway train gen, then val gen);
  - the same final-signal metric definition, by scoring the final output through the very
    same ContrastiveLoss.compute_signal_loss used for our model, so Val MAE / cosine / shape
    are computed identically and are directly comparable to SI Table S4.

Targets, in the amplitude frame our generator returns (input x in [0,1], clean signal y_s
scaled by 10, baseline y_b in [0,1]):
  - intermediate (baseline-corrected, still noisy) target = (x - y_b) * 10
  - final (clean) target                                 = y_s
so both stages operate in the same x10 frame and stage 2 performs a pure denoising step.

    python train_kazemzadeh.py --seed 0
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import train_ablation as T                     # shared config, data, helpers
from Pmodel import ContrastiveLoss, EarlyStopping
from kazemzadeh_model import CascadedUNet

SIG_SCALE = 10.0        # matches RamanDataGenerator's target scaling
W_INTER = 1.0           # weight on the intermediate (baseline-corrected) supervision
WEIGHT_DECAY = 1e-3     # L2 kernel regulariser, 0.001 in the source paper
LEARNING_RATE = 2e-4    # 2e-4 in the source paper (vs 1e-4 for our model)


def signal_metrics(criterion, pred, target):
    """(loss, mae, cos, shape) via the SAME machinery used to score our model.
    loss keeps its graph; the scalar metrics are detached for logging."""
    loss, mae, cos, shape = criterion.compute_signal_loss(pred, target, apply_shape=True)
    return loss, float(mae.detach()), float(cos.detach()), float(shape.detach())


def train(args):
    tag = args.tag or f"kazemzadeh_seed{args.seed}"
    os.makedirs(T.RUNS_DIR, exist_ok=True)
    os.makedirs(T.MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(T.RUNS_DIR, f"{tag}_best.pt")
    hist_csv = os.path.join(T.RUNS_DIR, f"{tag}_history.csv")
    meta_path = os.path.join(T.RUNS_DIR, f"{tag}_meta.json")
    plot_path = os.path.join(T.RUNS_DIR, f"{tag}_loss.png")
    model_name = f"Compare_kazemzadeh_cascadedUNet_seed{args.seed}.pth"
    model_path = os.path.join(T.MODELS_DIR, model_name)

    print("=" * 78)
    print(f"  CASCADED U-NET (Kazemzadeh et al. 2022)   seed={args.seed}   tag={tag}")
    print(f"  device: {T.DEVICE}")
    print("=" * 78)

    # --- data: identical stream and val set to our own runs -----------------
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
    model = CascadedUNet().to(T.DEVICE)
    n_total, _ = T.count_parameters(model)
    print(f"Parameters: {n_total:,}  (two U-Nets; cf. our 5.17M single dual-branch)")

    latency = T.benchmark_inference(model, train_gen.n_points, T.DEVICE)
    print(f"Latency: {latency['batch1_ms']:.3f} ms/spectrum (b1), "
          f"{latency['plate96_ms']:.3f} ms/96-well plate")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=T.SCHED_PATIENCE,
        factor=T.SCHED_FACTOR, min_lr=T.SCHED_MIN_LR)
    early = EarlyStopping(patience=T.ES_PATIENCE, start_epoch=T.ES_START_EPOCH,
                          delta=T.ES_DELTA, verbose=True, path=ckpt_path)
    # scoring machinery only -- construction weights are irrelevant to compute_signal_loss
    criterion = ContrastiveLoss()

    hist = {"train_loss": [], "val_loss": [], "val_mae": [], "val_cos": [], "val_shape": []}
    epoch_times = []
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        te = time.perf_counter()
        model.train()
        run = 0.0
        for x, y_s, y_b in train_dl:
            x, y_s, y_b = x.to(T.DEVICE), y_s.to(T.DEVICE), y_b.to(T.DEVICE)
            inter_target = (x - y_b) * SIG_SCALE           # baseline-corrected, noisy

            optimizer.zero_grad()
            final, inter, _base = model(x)
            loss_final, _, _, _ = signal_metrics(criterion, final, y_s)
            loss_inter = F.mse_loss(inter, inter_target)
            loss = loss_final + W_INTER * loss_inter
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=T.GRAD_CLIP_NORM)
            optimizer.step()
            run += loss.item()
        hist["train_loss"].append(run / len(train_dl))

        model.eval()
        vl = vmae = vcos = vsh = 0.0
        with torch.no_grad():
            for x, y_s, y_b in val_dl:
                x, y_s, y_b = x.to(T.DEVICE), y_s.to(T.DEVICE), y_b.to(T.DEVICE)
                final, inter, _base = model(x)
                lf, mae, cos, sh = signal_metrics(criterion, final, y_s)
                vl += lf.item(); vmae += mae; vcos += cos; vsh += sh
        nb = len(val_dl)
        vmae /= nb; vcos /= nb; vsh /= nb
        hist["val_loss"].append(vl / nb); hist["val_mae"].append(vmae)
        hist["val_cos"].append(vcos); hist["val_shape"].append(vsh)

        scheduler.step(vmae)
        epoch_times.append(time.perf_counter() - te)
        print(f"[kaz] Epoch {epoch+1}/{args.epochs} | LR {optimizer.param_groups[0]['lr']*1e4:.4f} "
              f"| Val MAE {vmae:.4f} | Val Cos {vcos:.4f} | Val Shape {vsh:.4f} | {epoch_times[-1]:.1f}s")

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
            x, y_s, y_b = x.to(T.DEVICE), y_s.to(T.DEVICE), y_b.to(T.DEVICE)
            final, _i, _b = model(x)
            _, mae, cos, sh = signal_metrics(criterion, final, y_s)
            r_mae += mae; r_cos += cos; r_sh += sh
    nb = len(val_dl)
    restored = {"val_mae": r_mae/nb, "val_cos": r_cos/nb, "val_shape": r_sh/nb}
    print(f"Restored-checkpoint metrics | Val MAE {restored['val_mae']:.6f} | "
          f"Val Cos {restored['val_cos']:.6f} | Val Shape {restored['val_shape']:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

    ep = range(1, len(hist["train_loss"]) + 1)
    np.savetxt(hist_csv,
               np.column_stack((ep, hist["train_loss"], hist["val_loss"],
                                hist["val_mae"], hist["val_cos"], hist["val_shape"], epoch_times)),
               delimiter=",",
               header="epoch,train_loss,val_loss,val_mae,val_cos,val_shape,epoch_seconds",
               comments="")

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(ep, hist["val_mae"], label="Val MAE")
    plt.plot(ep, hist["val_cos"], label="Val cosine loss")
    plt.plot(ep, hist["val_shape"], label="Val shape loss")
    plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Cascaded U-Net (Kazemzadeh) -- seed {args.seed}")
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(plot_path); plt.close()

    best_ix = int(np.argmin(hist["val_mae"]))
    now = datetime.now()
    meta = {
        "tag": tag, "model": "kazemzadeh_cascaded_unet",
        "reference": "Kazemzadeh et al., Anal. Chem. 2022, 94, 12907-12918",
        "variant": "cascaded U-Net (Fig. 3(b2))",
        "seed": args.seed, "device": str(T.DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "params_total": n_total, "latency": latency,
        "epochs_run": len(epoch_times),
        "mean_epoch_seconds": round(float(np.mean(epoch_times)), 3),
        "total_train_hours": round(total_time/3600, 4),
        "restored_val_mae": round(restored["val_mae"], 6),
        "restored_val_cos": round(restored["val_cos"], 6),
        "restored_val_shape": round(restored["val_shape"], 6),
        "argmin_epoch": best_ix + 1,
        "argmin_val_mae": round(hist["val_mae"][best_ix], 6),
        "model_path": model_path,
        "config": {"batch_size": args.batch_size, "learning_rate": LEARNING_RATE,
                   "weight_decay": WEIGHT_DECAY, "w_inter": W_INTER,
                   "sig_scale": SIG_SCALE,
                   "train_snr": [T.TRAIN_MIN_SNR, T.TRAIN_MAX_SNR],
                   "val_snr": [T.VAL_MIN_SNR, T.VAL_MAX_SNR],
                   "val_n_samples": T.VAL_N_SAMPLES},
        "timestamp": now.isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {meta_path}")

    # append to the shared experiment log (same 21-column schema)
    log = {
        "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S"),
        "Comment": (f"COMPARISON kazemzadeh cascaded-UNet seed={args.seed} "
                    f"params={n_total} epoch_s={meta['mean_epoch_seconds']} "
                    f"lat_plate96_ms={latency['plate96_ms']} [metrics=restored-checkpoint]"),
        "Model_Name": model_name, "Best_Epoch": best_ix + 1,
        "Val_MAE": round(restored["val_mae"], 6), "Val_Cos": round(restored["val_cos"], 6),
        "Val_Shape": round(restored["val_shape"], 6),
        "w_mae": 1.0, "w_cos": 1.0, "w_shape": 0.0, "w_signal": 1.0, "w_base": W_INTER,
        "w_consist": 0.0, "w_smooth": 0.0, "w_curve": 0.0, "w_mid": 0.0, "w_deep": 0.0,
        "w_ortho": 0.0, "Learning_Rate": LEARNING_RATE, "Batch_Size": args.batch_size,
    }
    exists = os.path.isfile(T.LOG_FILE)
    with open(T.LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log.keys()))
        if not exists:
            w.writeheader()
        w.writerow(log)
    print(f"Logged to {T.LOG_FILE}")

    print("\n" + "=" * 78)
    print(f"  RESULT  cascaded U-Net  seed={args.seed}  (saved-checkpoint metrics)")
    print(f"    Val MAE   : {restored['val_mae']:.6f}")
    print(f"    Val Cos   : {restored['val_cos']:.6f}")
    print(f"    Val Shape : {restored['val_shape']:.6f}")
    print(f"    params    : {n_total:,}")
    print(f"    s/epoch   : {meta['mean_epoch_seconds']}")
    print("=" * 78)
    return model, hist, meta


def parse_args():
    p = argparse.ArgumentParser(description="Cascaded U-Net comparator (Kazemzadeh 2022).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=T.BATCH_SIZE)
    p.add_argument("--tag", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
