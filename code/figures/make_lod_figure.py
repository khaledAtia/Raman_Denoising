"""SI figure: response, precision and detection limit on the synthetic distribution."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(BASE, "runs", "detection_limit.json")))
FIGS = os.path.join(BASE, "paper_revised", "figures")

RAMP = {"0.1": "#9ecae1", "0.05": "#4292c6", "0.025": "#08519c"}
fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=150)

for key, rows in d["curves"].items():
    cond, sig = key.rsplit("_", 1)
    if cond != "intermediate":
        continue
    a = np.array([r["amp"] for r in rows]); m = np.array([r["mean"] for r in rows])
    s = np.array([r["sd"] for r in rows])
    ax[0].errorbar(a, m, yerr=s, marker="o", ms=5, lw=1.6, capsize=3,
                   color=RAMP[sig], label=f"noise sd {sig}", zorder=3)
    ax[1].plot(a[1:], [r["rsd"] for r in rows[1:]], marker="o", ms=5, lw=1.6,
               color=RAMP[sig], label=f"noise sd {sig}", zorder=3)

lod = d["summary"]["intermediate_0.05"]["lod_amp"]
loq = d["summary"]["intermediate_0.05"]["loq_amp"]
for a_ in ax:
    a_.axvline(lod, color="0.5", ls="--", lw=1.0, zorder=1)
    a_.axvline(loq, color="0.5", ls=":", lw=1.0, zorder=1)
    for s_ in ("top", "right"):
        a_.spines[s_].set_visible(False)
    a_.grid(axis="y", color="0.92", lw=0.7); a_.set_axisbelow(True)
    a_.set_xlabel("Band amplitude (normalised units)")

ax[0].annotate("LOD", xy=(lod, 1.0), xycoords=("data", "axes fraction"),
               xytext=(2, -12), textcoords="offset points", fontsize=8, color="0.35")
ax[0].annotate("LOQ", xy=(loq, 1.0), xycoords=("data", "axes fraction"),
               xytext=(2, -12), textcoords="offset points", fontsize=8, color="0.35")
ax[0].set_ylabel("$I_{\mathrm{deep}}$")
ax[0].set_title("(a)  Response, mean $\pm$ 1 SD over 200 realisations",
                fontsize=10, loc="left")
ax[0].legend(frameon=False, fontsize=9)
ax[1].set_ylabel("Relative standard deviation (%)")
ax[1].set_yscale("log")
ax[1].set_title("(b)  Precision against amplitude", fontsize=10, loc="left")
ax[1].axhline(10, color="0.55", lw=0.9, ls="-.", zorder=1)
ax[1].annotate("10%", xy=(0.02, 10), fontsize=8, color="0.4", va="bottom")
ax[1].legend(frameon=False, fontsize=9)

fig.tight_layout()
out = os.path.join(FIGS, "detection_limit.png")
fig.savefig(out); print("wrote", out)
