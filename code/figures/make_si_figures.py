"""
Build the Supporting Information figures that can be drawn from results already on disk.

Three quantities are currently reported only as tables, and each is more readable as a
figure:

    convergence   validation error against epoch for every ablation arm. Establishes
                  visually that every arm converged, that the ordering between arms is
                  stable rather than an endpoint accident, and that removing a component
                  slows convergence -- a claim the tables cannot show at all.

    ablation      degradation of each metric when a component is removed, as a bar chart.
                  The ordering of the components by importance is immediate.

    mechanism     the analytic curvature contrast and the measured absorbed fraction, on a
                  common width axis. This is the causal claim of Section S7.3, and putting
                  the prediction and the measurement on one axis is the whole argument.

No GPU is required; everything is read from runs/*.json and runs/*_history.csv.

    python make_si_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
FIGS = os.path.join(BASE, "paper_revised", "figures")
os.makedirs(FIGS, exist_ok=True)

DISP = 1.8462  # cm-1 per point


def hist(tag):
    f = os.path.join(RUNS, f"{tag}_history.csv")
    return np.genfromtxt(f, delimiter=",", names=True) if os.path.isfile(f) else None


def meta(tag):
    f = os.path.join(RUNS, f"{tag}_meta.json")
    return json.load(open(f)) if os.path.isfile(f) else None


# ===========================================================================
# 1. CONVERGENCE
# ===========================================================================
ARMS = [
    ("rk4_seed0",              "Full model",                 "black",      "-",  2.0),
    ("rk4_nosquelch_seed0",    "no squelch gate",            "#d62728",    "-",  1.1),
    ("rk4_encoder-only_seed0", "no baseline input to gate",  "#ff7f0e",    "-",  1.1),
    ("rk4_nosigmoid_seed0",    "no sigmoid gating",          "#2ca02c",    "-",  1.1),
    ("rk4_noortho_seed0",      "no orthogonality loss",      "#9467bd",    "-",  1.1),
    ("rk4_noaux_seed0",        "no auxiliary supervision",   "#8c564b",    "-",  1.1),
    ("rk4_concat_seed0",       "gate input concatenated",    "#17becf",    "--", 1.0),
    ("plain_seed0",            "plain residual encoder",     "#1f77b4",    "-",  1.4),
]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), dpi=150)
present = []
for tag, lab, col, ls, lw in ARMS:
    h = hist(tag)
    if h is None:
        continue
    present.append(lab)
    e = np.arange(1, len(h["val_mae"]) + 1)
    ax[0].plot(e, h["val_mae"], color=col, ls=ls, lw=lw, label=lab)
    ax[1].plot(e, h["val_mae"], color=col, ls=ls, lw=lw)

ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Validation MAE")
ax[0].set_yscale("log"); ax[0].grid(alpha=0.3)
ax[0].set_title("(a) Full training history", fontsize=10)
ax[0].legend(fontsize=7, loc="upper right")

ax[1].set_xlim(400, None); ax[1].set_ylim(0.024, 0.040)
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Validation MAE")
ax[1].grid(alpha=0.3)
ax[1].set_title("(b) Detail after epoch 400", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figureS_convergence.png"))
plt.close()
print(f"convergence figure: {len(present)} arms -> figureS_convergence.png")

# ===========================================================================
# 2. MODULE ABLATION BAR CHART
# ===========================================================================
ctl = meta("rk4_seed0")
BARS = [
    ("rk4_nosquelch_seed0",    "terminal squelch gate"),
    ("rk4_encoder-only_seed0", "baseline input to the gate"),
    ("rk4_nosigmoid_seed0",    "sigmoid gating"),
    ("rk4_noortho_seed0",      "latent orthogonality loss"),
    ("rk4_noaux_seed0",        "auxiliary deep supervision"),
]
labels, d_mae, d_cos = [], [], []
for tag, lab in BARS:
    m = meta(tag)
    if m is None:
        continue
    labels.append(lab)
    d_mae.append((m["restored_val_mae"] - ctl["restored_val_mae"]) / ctl["restored_val_mae"] * 100)
    d_cos.append((m["restored_val_cos"] - ctl["restored_val_cos"]) / ctl["restored_val_cos"] * 100)

order = np.argsort(d_mae)
labels = [labels[i] for i in order]
d_mae = [d_mae[i] for i in order]
d_cos = [d_cos[i] for i in order]

y = np.arange(len(labels)); h = 0.38
fig, ax = plt.subplots(figsize=(7.2, 0.72 * len(labels) + 1.6), dpi=150)
ax.barh(y + h / 2, d_mae, h, color="#c44e52", label="mean absolute error")
ax.barh(y - h / 2, d_cos, h, color="#4c72b0", label="cosine loss")
for i, (a, b) in enumerate(zip(d_mae, d_cos)):
    ax.text(a + 0.25, i + h / 2, f"+{a:.1f}%", va="center", fontsize=8)
    ax.text(b + 0.25, i - h / 2, f"+{b:.1f}%", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Increase in validation error when the component is removed (\\%)")
ax.axvline(0, color="k", lw=1)
ax.set_xlim(0, max(d_cos + d_mae) * 1.22)
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figureS_ablation.png"))
plt.close()
print(f"ablation bar chart: {len(labels)} components -> figureS_ablation.png")

# ===========================================================================
# 3. MECHANISM: curvature contrast (analytic) vs absorbed fraction (measured)
# ===========================================================================
b = np.load(os.path.join(BASE, "data", "baseline_data.npz"))["data"]
B = b[:, 1:].T
Bn = (B - B.min(1, keepdims=True)) / (B.max(1, keepdims=True) - B.min(1, keepdims=True) + 1e-12)
tv2_bg = float(np.abs(np.diff(Bn, 2, axis=1)).mean())


def pv(x, c, w, eta=0.9):
    sg = w / 2.0
    dd = (x - c) / sg
    return eta / (1 + dd ** 2) + (1 - eta) * np.exp(-np.log(2) * dd ** 2)


x = np.arange(864)
widths = np.arange(3, 31)
contrast = np.array([np.abs(np.diff(pv(x, 432, w), 2)).mean() / tv2_bg for w in widths])
fwhm = widths * DISP

mb = json.load(open(os.path.join(RUNS, "rk4_seed0_massbalance.json")))
mb_f = np.array([r["fwhm_cm"] for r in mb["rows"]])
mb_a = np.array([r["absorbed_mean"] for r in mb["rows"]])

fig, ax1 = plt.subplots(figsize=(6.8, 4.3), dpi=150)
ax1.plot(fwhm, contrast, "o-", ms=3.5, color="#4c72b0",
         label="curvature contrast, peak / background (predicted)")
ax1.axhline(1.0, color="#4c72b0", ls=":", lw=1)
ax1.set_xlabel("Band FWHM (cm$^{-1}$)")
ax1.set_ylabel("Curvature contrast", color="#4c72b0")
ax1.tick_params(axis="y", labelcolor="#4c72b0")
ax1.set_yscale("log"); ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(mb_f, mb_a * 100, "s-", ms=3.5, color="#c44e52",
         label="area absorbed by the baseline branch (measured)")
ax2.set_ylabel("Absorbed fraction (\\%)", color="#c44e52")
ax2.tick_params(axis="y", labelcolor="#c44e52")

l1, b1 = ax1.get_legend_handles_labels()
l2, b2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, b1 + b2, fontsize=8, loc="upper center")
plt.title("Predicted separability and measured absorption", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "figureS_mechanism.png"))
plt.close()
print("mechanism figure -> figureS_mechanism.png")
print(f"  contrast falls {contrast[0]:.1f} -> {contrast[-1]:.1f} as FWHM goes "
      f"{fwhm[0]:.1f} -> {fwhm[-1]:.1f} cm-1")


# ===========================================================================
# 4. LOSS-TERM BEHAVIOUR AT CONVERGENCE
# ===========================================================================
h = hist("rk4_seed0")
if h is not None:
    e = np.arange(1, len(h["val_mae"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1), dpi=150)

    ax[0].plot(e, h["train_ortho"], color="#9467bd", lw=1.3, label=r"orthogonality $L_{ortho}$")
    ax[0].plot(e, h["train_aux_s"], color="#8c564b", lw=1.3, label=r"auxiliary signal $L_s^{deep}$")
    ax[0].plot(e, h["val_mae"], color="#c44e52", lw=1.3, label=r"amplitude $L_{L1}$")
    ax[0].plot(e, h["val_cos"], color="#4c72b0", lw=1.3, label=r"cosine $L_{cos}$")
    ax[0].plot(e, h["val_shape"], color="#55a868", lw=1.3, label=r"shape $L_{shape}$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss term")
    ax[0].set_title("(a) Loss terms during training", fontsize=10)
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=0.3)

    terms = [(r"$L_s^{deep}$", h["train_aux_s"][-50:].mean(), 1.0),
             (r"$L_{L1}$",     h["val_mae"][-50:].mean(),     1.0),
             (r"$L_{cos}$",    h["val_cos"][-50:].mean(),     1.0),
             (r"$L_{shape}$",  h["val_shape"][-50:].mean(),   0.1),
             (r"$L_{ortho}$",  h["train_ortho"][-50:].mean(), 0.5)]
    lab = [t[0] for t in terms]
    raw = np.array([t[1] for t in terms])
    wtd = np.array([t[1] * t[2] for t in terms])
    y = np.arange(len(lab)); bh = 0.38
    ax[1].barh(y + bh / 2, raw, bh, color="#bbbbbb", label="unweighted")
    ax[1].barh(y - bh / 2, wtd, bh, color="#c44e52", label="after its weight")
    ax[1].set_yticks(y); ax[1].set_yticklabels(lab)
    ax[1].set_xscale("log"); ax[1].set_xlabel("Magnitude at convergence")
    ax[1].set_title("(b) Contribution of each term", fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "figureS_lossterms.png"))
    plt.close()
    print("loss-term figure -> figureS_lossterms.png")
    for l, r_, w_ in terms:
        print(f"   {l:<14} raw {r_:.6f}   weighted {r_*w_:.6f}")
