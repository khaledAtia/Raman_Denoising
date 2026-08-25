"""
Controlled ablation trainer: RK4 encoder block  vs.  plain pre-activation residual block.

Answers reviewer comments R1.1 / R2.2 / R3.7 ("the role of each key module should be shown
more clearly", "how about not adopt the auxiliary loss", "ablation results would confirm
these numbers are not randomly picked").

WHY ONE SCRIPT FOR BOTH ARMS
----------------------------
An ablation is only evidence if the two arms differ in exactly one variable. This script
trains either architecture through an identical code path, with identical hyperparameters,
an identical data stream and an identical validation set, selected by --arch.

    python train_ablation.py --arch rk4   --seed 0     # control  (Pmodel.py,          paper model)
    python train_ablation.py --arch plain --seed 0     # ablation (Pmodel_plainres.py, plain residual)

The data generator draws exclusively from numpy's global RNG (no torch RNG), while model
initialisation draws exclusively from torch's. Seeding both up front therefore gives the
two arms a bit-identical data stream even though their weight initialisations differ in
shape and count. The fixed 2000-sample validation set is likewise identical across arms.

Hyperparameters are copied verbatim from train2.py::train_model(), which produced the
checkpoint reported in the paper:
    models/Pmod_nosmoothBase_113_S20.0_B20.0_wm2.0_wd1.0_ortho0.5_wmae1.0_wcos1.0_wshape0.1.pth
Do not change them here -- that is the point of a control.

OUTPUTS (all namespaced by --tag, so nothing from a previous run is overwritten)
    models/Ablation_<tag>.pth              final weights (best-epoch weights restored)
    runs/<tag>_best.pt                     early-stopping checkpoint
    runs/<tag>_history.csv                 per-epoch losses
    runs/<tag>_meta.json                   arch, seed, param count, timings, final metrics
    runs/<tag>_dynamics.png                orthogonality / deep-supervision curves
    experiment_tracking_log.csv            one appended row (existing 21-column schema kept)

NOTE ON cuDNN: convolution autotuning is not bit-deterministic on GPU. Seeds make the runs
statistically comparable, not bit-identical. For a publishable table, run each arm over
>= 3 seeds and report mean +/- std -- see --seed and the aggregate hint at the bottom.
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")           # headless: never block on a GUI during a long run
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Loss + early stopping come from the ORIGINAL module in both arms, so the only thing
# --arch can possibly change is the encoder residual block.
from Pmodel import ContrastiveLoss, EarlyStopping
from Pmodel import AUSequentialUNet as RK4UNet
from Pmodel_plainres import AUSequentialUNet as PlainResUNet
from Pmodel_gate import AUSequentialUNet as GateUNet
from Pmodel_ablate import AUSequentialUNet as AblateUNet
from RamanDataGenerator import RamanDataGenerator, FixedRamanDataset


# ==========================================================================
# PATHS  (resolved relative to this file so the script is location-independent)
# ==========================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(BASE_DIR, "data", "baseline_data.npz")
NOISE_FILE    = os.path.join(BASE_DIR, "data", "noise_data.npz")
RUNS_DIR      = os.path.join(BASE_DIR, "runs")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
LOG_FILE      = os.path.join(BASE_DIR, "experiment_tracking_log.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# FROZEN CONFIG  -- verbatim from train2.py::train_model()
# ==========================================================================
BATCH_SIZE    = 64
LEARNING_RATE = 1e-4

GAMMA   = 1.0
KERNELS = [1, 1, 3]
BASE_LATENT_DIM = 16
USE_DERIVATIVES = False

# Data
TRAIN_EPOCH_SIZE = 100 * BATCH_SIZE      # 6400 spectra per epoch, regenerated each epoch
TRAIN_MIN_SNR, TRAIN_MAX_SNR = 4, 60
VAL_MIN_SNR,   VAL_MAX_SNR   = 4, 26
VAL_N_SAMPLES   = 2000
VAL_BLANK_RATIO = 0.0                    # training keeps the generator default blank_ratio

# Loss weights
W_SIGNAL, W_BASE = 20.0, 20.0
W_CONSIST        = 0.0
W_SMOOTH, W_CURVE = 0.2, 0.2
W_MAE, W_COS, W_SHAPE = 1.0, 1.0, 0.1
D_FACTOR = 0.0
W_MID, W_DEEP = 2.0, 1.0
W_ORTHO = 0.5

# Schedule
SCHED_PATIENCE, SCHED_FACTOR, SCHED_MIN_LR = 15, 0.5, 1e-8
ES_PATIENCE, ES_START_EPOCH, ES_DELTA = 100, 250, 1e-4
GRAD_CLIP_NORM = 1.0

ARCHITECTURES = {"rk4": RK4UNet, "plain": PlainResUNet}
GATE_MODES = ("subtract", "concat", "encoder-only")


def build_model(arch, gate, gate_sigmoid=True, squelch=True,
                base_latent_dim=BASE_LATENT_DIM, widths=(64, 128, 256, 512)):
    """
    Model selection.

    Everything at its default reproduces Pmodel.py, the model reported in the
    manuscript. Deviating from the defaults selects an ablation variant, and each
    variant differs from the control in exactly one respect.

      arch='plain'                -> Pmodel_plainres.py, the encoder-block ablation
      gate='concat'|'encoder-only'-> gate-input ablation
      gate_sigmoid=False          -> evidence passes ungated
      squelch=False               -> terminal squelch gate removed
      base_latent_dim != 16       -> bottleneck channel allocation
      widths != (64,...,512)      -> encoder width

    Architectural ablations hold the encoder at RK4, since that is the reported
    model; varying the encoder block at the same time would not be a
    single-variable comparison.
    """
    default_shape = (gate == "subtract" and gate_sigmoid and squelch
                     and base_latent_dim == BASE_LATENT_DIM
                     and tuple(widths) == (64, 128, 256, 512))

    if default_shape:
        # the two encoder-block arms, straight from their own modules
        return ARCHITECTURES[arch](
            n_channels=1, bilinear=False, gamma=GAMMA, kernels=KERNELS,
            base_latent_dim=BASE_LATENT_DIM, use_derivatives=USE_DERIVATIVES)

    if arch != "rk4":
        raise SystemExit(
            f"architectural ablations are only defined with --arch rk4 (got --arch {arch}). "
            "They hold the encoder fixed at the reported model; varying both at once "
            "would not be a single-variable comparison.")

    return AblateUNet(
        n_channels=1, bilinear=False, gamma=GAMMA, kernels=KERNELS,
        base_latent_dim=base_latent_dim, use_derivatives=USE_DERIVATIVES,
        hint_mode=gate, gate_sigmoid=gate_sigmoid, squelch=squelch,
        widths=tuple(widths))


def set_seed(seed):
    """Seed every RNG the training path touches."""
    random.seed(seed)
    np.random.seed(seed)             # <- the data generator draws from this one
    torch.manual_seed(seed)          # <- weight init draws from this one
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def benchmark_inference(model, n_points, device, n_warmup=10, n_iter=100):
    """
    Single-spectrum and full-plate latency. Feeds T14 (R3.4: 'little discussion about the
    inference performance of this model') at essentially zero extra cost, since the model
    is already built and on the device.
    """
    model.eval()
    results = {}
    for label, batch in (("batch1", 1), ("plate96", 96)):
        x = torch.randn(batch, 1, n_points, device=device)
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_iter
        results[f"{label}_ms"] = round(elapsed * 1e3, 4)
        results[f"{label}_ms_per_spectrum"] = round(elapsed * 1e3 / batch, 4)
    return results


def train(args):
    # Accept either a parsed Namespace (CLI use) or a bare arch string, so the file can be
    # launched straight from the VS Code Run button with train("plain") / train("rk4").
    if isinstance(args, str):
        args = make_config(arch=args)

    gate = getattr(args, "gate", "subtract")
    gate_sigmoid = not getattr(args, "no_gate_sigmoid", False)
    squelch = not getattr(args, "no_squelch", False)
    bld = getattr(args, "base_latent_dim", BASE_LATENT_DIM)
    widths = tuple(getattr(args, "widths", (64, 128, 256, 512)))
    w_ortho = 0.0 if getattr(args, "no_ortho", False) else W_ORTHO
    w_mid = 0.0 if getattr(args, "no_aux", False) else W_MID
    w_deep = 0.0 if getattr(args, "no_aux", False) else W_DEEP

    parts = []
    if gate != "subtract":              parts.append(gate)
    if not gate_sigmoid:                parts.append("nosigmoid")
    if not squelch:                     parts.append("nosquelch")
    if getattr(args, "no_ortho", False): parts.append("noortho")
    if getattr(args, "no_aux", False):   parts.append("noaux")
    if bld != BASE_LATENT_DIM:          parts.append(f"bld{bld}")
    if widths != (64, 128, 256, 512):   parts.append(f"w{widths[0]}")
    suffix = ("_" + "_".join(parts)) if parts else ""
    tag = args.tag or f"{args.arch}{suffix}_seed{args.seed}"

    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    ckpt_path   = os.path.join(RUNS_DIR, f"{tag}_best.pt")
    history_csv = os.path.join(RUNS_DIR, f"{tag}_history.csv")
    meta_path   = os.path.join(RUNS_DIR, f"{tag}_meta.json")
    plot_path   = os.path.join(RUNS_DIR, f"{tag}_dynamics.png")

    ks = "".join(map(str, KERNELS))
    model_name = (f"Ablation_{args.arch}{suffix}_{ks}_S{W_SIGNAL}_B{W_BASE}_wm{W_MID}_wd{W_DEEP}"
                  f"_ortho{W_ORTHO}_wmae{W_MAE}_wcos{W_COS}_wshape{W_SHAPE}_seed{args.seed}.pth")
    model_path = os.path.join(MODELS_DIR, model_name)

    print("=" * 78)
    print(f"  ABLATION RUN   arch={args.arch}   seed={args.seed}   tag={tag}")
    print(f"  encoder block: {'RK4SmoothedBlock (control)' if args.arch == 'rk4' else 'SmoothedResBlock (ablated)'}")
    _gate_desc = {"subtract": "E_l - B_l  (control)",
                  "concat": "[E_l || B_l]",
                  "encoder-only": "E_l  (baseline branch does not gate)"}[gate]
    print(f"  gate input   : hint = {_gate_desc}")
    print(f"  device: {DEVICE}")
    print("=" * 78)

    # --- 1. Seed BEFORE any data or model is built ---------------------------
    set_seed(args.seed)

    # --- 2. Data -------------------------------------------------------------
    print("Initializing data generators...")
    train_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE,
                                   epoch_size=TRAIN_EPOCH_SIZE,
                                   min_snr=TRAIN_MIN_SNR, max_snr=TRAIN_MAX_SNR)
    train_dataloader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True)

    temp_val_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE,
                                      min_snr=VAL_MIN_SNR, max_snr=VAL_MAX_SNR,
                                      blank_ratio=VAL_BLANK_RATIO)
    val_dataset = FixedRamanDataset(temp_val_gen, n_samples=VAL_N_SAMPLES)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Training on {DEVICE} with {train_gen.n_points} points per spectrum.")
    print(f"Split: dynamic train set ({len(train_gen)}/epoch, SNR {TRAIN_MIN_SNR}-{TRAIN_MAX_SNR}) | "
          f"fixed val set ({len(val_dataset)}, SNR {VAL_MIN_SNR}-{VAL_MAX_SNR})")

    # --- 3. Model ------------------------------------------------------------
    model = build_model(args.arch, gate, gate_sigmoid=gate_sigmoid, squelch=squelch,
                        base_latent_dim=bld, widths=widths).to(DEVICE)

    n_total, n_trainable = count_parameters(model)
    print(f"Parameters: {n_total:,} total / {n_trainable:,} trainable")

    latency = benchmark_inference(model, train_gen.n_points, DEVICE)
    print(f"Inference latency: {latency['batch1_ms']:.3f} ms/spectrum (batch 1), "
          f"{latency['plate96_ms']:.3f} ms/96-well plate "
          f"({latency['plate96_ms_per_spectrum']:.4f} ms/spectrum)")

    # --- 4. Optimiser / loss -------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHED_PATIENCE,
        factor=SCHED_FACTOR, min_lr=SCHED_MIN_LR)
    early_stopping = EarlyStopping(patience=ES_PATIENCE, start_epoch=ES_START_EPOCH,
                                   delta=ES_DELTA, verbose=True, path=ckpt_path)

    criterion = ContrastiveLoss(
        w_signal=W_SIGNAL, w_base=W_BASE, w_consist=W_CONSIST,
        w_smooth=W_SMOOTH, w_curve=W_CURVE, dip_factor=D_FACTOR,
        w_mae=W_MAE, w_cos=W_COS, w_shape=W_SHAPE,
        w_mid=w_mid, w_deep=w_deep, w_ortho=w_ortho)

    # --- 5. Training loop ----------------------------------------------------
    loss_history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_cos": [],
                    "val_shape": [], "train_aux_s": [], "train_aux_b": [], "train_ortho": []}
    epoch_times = []
    t_start = time.perf_counter()

    for epoch in range(args.epochs):
        t_epoch = time.perf_counter()

        # ---- train ----
        model.train()
        ep_loss = ep_aux_s = ep_aux_b = ep_ortho = 0.0

        for inputs, target_s, target_b in train_dataloader:
            inputs   = inputs.to(DEVICE)
            target_s = target_s.to(DEVICE)
            target_b = target_b.to(DEVICE)

            optimizer.zero_grad()
            pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)

            loss, t_components = criterion(
                pred_s, target_s, pred_b, target_b, inputs,
                pred_s_mid=pred_s_mid, pred_s_deep=pred_s_deep,
                pred_b_mid=pred_b_mid, pred_b_deep=pred_b_deep,
                x4_base=x4_base, x4_sig=x4_sig)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            ep_loss  += loss.item()
            ep_aux_s += t_components[7]     # deep signal
            ep_aux_b += t_components[8]     # deep baseline
            ep_ortho += t_components[9]     # orthogonality

        nb = len(train_dataloader)
        loss_history["train_loss"].append(ep_loss / nb)
        loss_history["train_aux_s"].append(ep_aux_s / nb)
        loss_history["train_aux_b"].append(ep_aux_b / nb)
        loss_history["train_ortho"].append(ep_ortho / nb)

        # ---- validate ----
        model.eval()
        v_loss_sum = v_mae = v_cos = v_shape = 0.0
        with torch.no_grad():
            for inputs, target_s, target_b in val_dataloader:
                inputs   = inputs.to(DEVICE)
                target_s = target_s.to(DEVICE)
                target_b = target_b.to(DEVICE)

                pred_s, pred_b, _, _, _, _, x4_base, x4_sig = model(inputs)
                v_loss, v_components = criterion(
                    pred_s, target_s, pred_b, target_b, inputs,
                    x4_base=x4_base, x4_sig=x4_sig)

                v_loss_sum += v_loss.item()
                v_mae   += v_components[0]
                v_cos   += v_components[1]
                v_shape += v_components[2]

        nv = len(val_dataloader)
        avg_val_loss  = v_loss_sum / nv
        avg_val_mae   = v_mae / nv
        avg_val_cos   = v_cos / nv
        avg_val_shape = v_shape / nv

        loss_history["val_loss"].append(avg_val_loss)
        loss_history["val_mae"].append(avg_val_mae)
        loss_history["val_cos"].append(avg_val_cos)
        loss_history["val_shape"].append(avg_val_shape)

        # ---- log / schedule / stop ----
        scheduler.step(avg_val_mae)
        epoch_times.append(time.perf_counter() - t_epoch)

        print(f"[{args.arch}] Epoch {epoch + 1}/{args.epochs} | "
              f"LR {optimizer.param_groups[0]['lr'] * 1e4:.4f} | "
              f"Val MAE: {avg_val_mae:.4f} | Val Cos: {avg_val_cos:.4f} | "
              f"Val Shape: {avg_val_shape:.4f} | Ortho: {loss_history['train_ortho'][-1]:.4f} | "
              f"AuxS: {loss_history['train_aux_s'][-1]:.4f} | {epoch_times[-1]:.1f}s")

        early_stopping(avg_val_mae, model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    total_time = time.perf_counter() - t_start
    print(f"Training complete in {total_time / 3600:.2f} h "
          f"({np.mean(epoch_times):.1f} s/epoch over {len(epoch_times)} epochs).")

    # --- 6. Restore best weights and save ------------------------------------
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print("Loaded best model weights.")
    else:
        print("WARNING: no early-stopping checkpoint found "
              f"(run ended before start_epoch={ES_START_EPOCH}); saving final-epoch weights.")

    # Re-evaluate the RESTORED weights on the fixed validation set.
    #
    # Why this is necessary: EarlyStopping only records an improvement when the
    # metric beats its best by more than `delta` (1e-4). The raw argmin of the
    # val_mae history can therefore land on a later epoch than the one whose
    # weights were actually checkpointed. Reporting the argmin alongside the
    # checkpointed weights describes a model that was never saved. These numbers
    # go straight into the SI ablation table, so they must describe the weights
    # on disk. The validation set is fixed, so this pass is deterministic.
    model.eval()
    r_loss = r_mae = r_cos = r_shape = 0.0
    with torch.no_grad():
        for inputs, target_s, target_b in val_dataloader:
            inputs   = inputs.to(DEVICE)
            target_s = target_s.to(DEVICE)
            target_b = target_b.to(DEVICE)
            pred_s, pred_b, _, _, _, _, x4_base, x4_sig = model(inputs)
            v_loss, v_comp = criterion(pred_s, target_s, pred_b, target_b, inputs,
                                       x4_base=x4_base, x4_sig=x4_sig)
            r_loss  += v_loss.item()
            r_mae   += v_comp[0]
            r_cos   += v_comp[1]
            r_shape += v_comp[2]
    nvr = len(val_dataloader)
    restored = {"val_loss": r_loss / nvr, "val_mae": r_mae / nvr,
                "val_cos": r_cos / nvr, "val_shape": r_shape / nvr}
    print(f"Restored-checkpoint metrics | Val MAE: {restored['val_mae']:.6f} | "
          f"Val Cos: {restored['val_cos']:.6f} | Val Shape: {restored['val_shape']:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # --- 7. History CSV ------------------------------------------------------
    trained_epochs = range(1, len(loss_history["train_loss"]) + 1)
    np.savetxt(history_csv,
               np.column_stack((trained_epochs,
                                loss_history["train_ortho"], loss_history["train_aux_s"],
                                loss_history["val_mae"], loss_history["val_cos"],
                                loss_history["val_shape"], loss_history["train_loss"],
                                loss_history["val_loss"], epoch_times)),
               delimiter=",",
               header="epoches,train_ortho,train_aux_s,val_mae,val_cos,val_shape,train_loss,val_loss,epoch_seconds",
               comments="")
    print(f"History saved to: {history_csv}")

    # --- 8. Dynamics plot ----------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(trained_epochs, loss_history["train_ortho"], label="Latent Orthogonality Penalty",
             color="purple", linewidth=2)
    plt.plot(trained_epochs, loss_history["train_aux_s"], label="Deep Supervision (Signal)",
             color="crimson", linewidth=2, linestyle="--")
    plt.plot(trained_epochs, loss_history["train_aux_b"], label="Deep Supervision (Baseline)",
             color="dodgerblue", linewidth=2, linestyle="--")
    plt.title(f"Latent Space & Deep Supervision Dynamics ({args.arch}, seed {args.seed})",
              fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # --- 9. Metadata sidecar -------------------------------------------------
    best_val_mae  = min(loss_history["val_mae"])
    best_epoch_ix = loss_history["val_mae"].index(best_val_mae)
    now = datetime.now()

    meta = {
        "tag": tag,
        "arch": args.arch,
        "gate": gate,
        "gate_sigmoid": gate_sigmoid,
        "squelch": squelch,
        "base_latent_dim": bld,
        "widths": list(widths),
        "w_ortho_used": w_ortho,
        "w_mid_used": w_mid,
        "w_deep_used": w_deep,
        "encoder_block": "RK4SmoothedBlock" if args.arch == "rk4" else "SmoothedResBlock",
        "module": ("Pmodel_gate" if gate != "subtract"
                   else ("Pmodel" if args.arch == "rk4" else "Pmodel_plainres")),
        "seed": args.seed,
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "params_total": n_total,
        "params_trainable": n_trainable,
        "latency": latency,
        "epochs_run": len(epoch_times),
        "mean_epoch_seconds": round(float(np.mean(epoch_times)), 3),
        "total_train_hours": round(total_time / 3600, 4),

        # AUTHORITATIVE -- metrics of the weights actually saved to model_path.
        # Use these for the SI ablation table.
        "restored_val_mae": round(restored["val_mae"], 6),
        "restored_val_cos": round(restored["val_cos"], 6),
        "restored_val_shape": round(restored["val_shape"], 6),

        # Raw argmin over the training history. May fall on a LATER epoch than the
        # checkpoint, because EarlyStopping applies a delta=1e-4 improvement
        # threshold. Kept for reference only -- do not report these as the model's
        # performance.
        "argmin_epoch": best_epoch_ix + 1,
        "argmin_val_mae": round(best_val_mae, 6),
        "argmin_val_cos": round(loss_history["val_cos"][best_epoch_ix], 6),
        "argmin_val_shape": round(loss_history["val_shape"][best_epoch_ix], 6),
        "model_path": model_path,
        "config": {
            "batch_size": args.batch_size, "learning_rate": LEARNING_RATE,
            "kernels": KERNELS, "base_latent_dim": BASE_LATENT_DIM,
            "use_derivatives": USE_DERIVATIVES, "gamma": GAMMA,
            "train_snr": [TRAIN_MIN_SNR, TRAIN_MAX_SNR],
            "val_snr": [VAL_MIN_SNR, VAL_MAX_SNR],
            "train_epoch_size": TRAIN_EPOCH_SIZE, "val_n_samples": VAL_N_SAMPLES,
            "w_signal": W_SIGNAL, "w_base": W_BASE, "w_consist": W_CONSIST,
            "w_smooth": W_SMOOTH, "w_curve": W_CURVE, "dip_factor": D_FACTOR,
            "w_mae": W_MAE, "w_cos": W_COS, "w_shape": W_SHAPE,
            "w_mid": W_MID, "w_deep": W_DEEP, "w_ortho": W_ORTHO,
        },
        "timestamp": now.isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to: {meta_path}")

    # --- 10. Append to the shared experiment log -----------------------------
    # Schema is pinned to the existing 21 columns so the historical file stays readable;
    # arch/seed/params/timing live in Comment and in the JSON sidecar.
    auto_comment = (f"ABLATION arch={args.arch} variant={suffix or 'control'} seed={args.seed} "
                    f"params={n_total} epoch_s={meta['mean_epoch_seconds']} "
                    f"lat_b1_ms={latency['batch1_ms']}")
    if args.comment:
        auto_comment = f"{args.comment} | {auto_comment}"

    log_data = {
        "Date": now.strftime("%Y-%m-%d"), "Time": now.strftime("%H:%M:%S"),
        "Comment": auto_comment, "Model_Name": model_name,
        "Best_Epoch": best_epoch_ix + 1,
        # Metrics of the RESTORED checkpoint, not the raw argmin -- see meta.json.
        "Val_MAE": round(restored["val_mae"], 6),
        "Val_Cos": round(restored["val_cos"], 6),
        "Val_Shape": round(restored["val_shape"], 6),
        "w_mae": W_MAE, "w_cos": W_COS, "w_shape": W_SHAPE,
        "w_signal": W_SIGNAL, "w_base": W_BASE, "w_consist": W_CONSIST,
        "w_smooth": W_SMOOTH, "w_curve": W_CURVE,
        "w_mid": W_MID, "w_deep": W_DEEP, "w_ortho": W_ORTHO,
        "Learning_Rate": LEARNING_RATE, "Batch_Size": args.batch_size,
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    print(f"Experiment logged to {LOG_FILE}")

    print("\n" + "=" * 78)
    print(f"  RESULT  arch={args.arch} seed={args.seed}   (saved-checkpoint metrics)")
    print(f"    Val MAE         : {restored['val_mae']:.6f}")
    print(f"    Val Cos         : {restored['val_cos']:.6f}")
    print(f"    Val Shape       : {restored['val_shape']:.6f}")
    print(f"    parameters      : {n_total:,}")
    print(f"    epochs run      : {len(epoch_times)}")
    print(f"    s/epoch         : {meta['mean_epoch_seconds']}")
    print(f"    [ref] argmin epoch {best_epoch_ix + 1}, argmin Val MAE {best_val_mae:.6f}")
    print("=" * 78)

    return model, loss_history, meta


def make_config(arch, seed=0, epochs=2000, batch_size=BATCH_SIZE, tag=None, comment="",
                gate="subtract", no_gate_sigmoid=False, no_squelch=False,
                no_ortho=False, no_aux=False, base_latent_dim=BASE_LATENT_DIM,
                widths=(64, 128, 256, 512)):
    """Build the same config object argparse produces, for non-CLI (VS Code Run button) use."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}, got {arch!r}")
    return argparse.Namespace(arch=arch, seed=seed, epochs=epochs,
                              batch_size=batch_size, tag=tag, comment=comment, gate=gate,
                              no_gate_sigmoid=no_gate_sigmoid, no_squelch=no_squelch,
                              no_ortho=no_ortho, no_aux=no_aux,
                              base_latent_dim=base_latent_dim, widths=widths)


def parse_args():
    p = argparse.ArgumentParser(
        description="Controlled RK4-vs-plain-residual encoder ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", choices=sorted(ARCHITECTURES), default="plain",
                   help="'rk4' = Pmodel.py control (paper model); "
                        "'plain' = Pmodel_plainres.py ablation")
    p.add_argument("--gate", choices=GATE_MODES, default="subtract",
                   help="what the cross-branch gate receives as its hint. "
                        "'subtract' = E_l - B_l (reported model); 'concat' = [E_l||B_l]; "
                        "'encoder-only' = E_l. Requires --arch rk4 when not 'subtract'. "
                        "Answers R3.1.")
    p.add_argument("--no-gate-sigmoid", action="store_true",
                   help="pass the gate evidence ungated (removes the sigmoid gating)")
    p.add_argument("--no-squelch", action="store_true",
                   help="remove the terminal spatial squelch gate")
    p.add_argument("--no-ortho", action="store_true",
                   help="set the latent orthogonality weight to zero")
    p.add_argument("--no-aux", action="store_true",
                   help="set both auxiliary deep-supervision weights to zero")
    p.add_argument("--base-latent-dim", type=int, default=BASE_LATENT_DIM,
                   help="channels routed to the baseline branch at the bottleneck")
    p.add_argument("--widths", type=int, nargs=4, default=[64, 128, 256, 512],
                   metavar=("W0", "W1", "W2", "W3"),
                   help="encoder channel schedule; entries must double")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds numpy (data stream) and torch (weight init)")
    p.add_argument("--epochs", type=int, default=2000,
                   help="max epochs; early stopping normally ends the run well before this")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--tag", default=None,
                   help="output namespace (default: <arch>_seed<seed>)")
    p.add_argument("--comment", default="",
                   help="free-text note prepended to the experiment log row")
    return p.parse_args()


if __name__ == "__main__":
    # VS Code Run button (no CLI args) -> edit this line: train("plain") or train("rk4").
    # Terminal / F5 launch configs pass --arch and override it via parse_args().
    train(parse_args())

# ---------------------------------------------------------------------------
# To produce the ablation table for the revision, run both arms over 3 seeds:
#
#   for s in 0 1 2; do
#     python train_ablation.py --arch rk4   --seed $s
#     python train_ablation.py --arch plain --seed $s
#   done
#
# then aggregate runs/*_meta.json into mean +/- std of best_val_mae / val_cos_at_best,
# alongside params_total and mean_epoch_seconds. Reporting a single seed per arm invites
# exactly the objection R3 already raised about Figure 10 (comment 6): one point per
# condition is not evidence.
# ---------------------------------------------------------------------------
