"""
Run the remaining ablation and comparison studies.

The studies are organised into three groups, each answering a different question and each
runnable on its own so that the work can be spread over several sessions:

    module      does each architectural component earn its place?          4 runs, ~14 h
    bottleneck  is the asymmetric 16/496 channel split the right choice?   3 runs, ~10.5 h
    width       is the 64/128/256/512 encoder the right size?              2 runs, ~7 h

A fourth group, "comparison", holds the cascaded-U-Net comparator; it is already complete
and is listed only so that the suite reports it as such.

    python run_ablation_suite.py --dry-run             # list everything, with estimates
    python run_ablation_suite.py --group module        # run one group
    python run_ablation_suite.py --group bottleneck width
    python run_ablation_suite.py                       # run every outstanding study

Runs are executed sequentially in subprocesses: a failure in one does not take down the
rest, and anything whose meta.json already exists is skipped, so the suite is safe to stop
and restart. Every study differs from the reported model in exactly one respect, and all
share the seed, the training stream and the frozen validation set.

Note that the loss-weight sensitivity table requires no training at all; it is compiled
from the historical rows of experiment_tracking_log.csv.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "runs")
PY = sys.executable

# Wall-clock estimate per run, from the completed arms (726 epochs at ~17 s).
HOURS_PER_RUN = 3.5

GROUPS = {
    "module":     "module ablations -- does each component earn its place?",
    "bottleneck": "bottleneck channel allocation -- is 16/496 the right split?",
    "width":      "encoder width -- is 64/128/256/512 the right size?",
    "comparison": "comparison against a published method",
}


class Study:
    """One training run: how to launch it, and how to tell whether it is already done."""

    def __init__(self, key, group, label, tag_fn, argv_fn, script="train_ablation.py"):
        self.key = key
        self.group = group
        self.label = label
        self.tag_fn = tag_fn
        self.argv_fn = argv_fn
        self.script = script

    def tag(self, seed):
        return self.tag_fn(seed)

    def meta(self, seed):
        return os.path.join(RUNS, f"{self.tag(seed)}_meta.json")

    def done(self, seed):
        return os.path.isfile(self.meta(seed))

    def argv(self, seed):
        return [PY, os.path.join(BASE, self.script)] + self.argv_fn(seed)


def _ab(extra_args, tag_suffix):
    """Helper for train_ablation.py studies: rk4 encoder, one thing changed."""
    return (lambda s: f"rk4_{tag_suffix}_seed{s}",
            lambda s: ["--arch", "rk4", "--seed", str(s)] + extra_args)


STUDIES = []

# ---- GROUP 1: module ablations ---------------------------------------------
# Each removes one component and leaves everything else untouched.
for key, label, args_, suffix in [
    ("sigmoid", "no sigmoid gating (hint evidence passes ungated)",
     ["--no-gate-sigmoid"], "nosigmoid"),
    ("squelch", "no terminal squelch gate (inter-peak floor not suppressed)",
     ["--no-squelch"], "nosquelch"),
    ("ortho", "no latent orthogonality loss (w_ortho = 0)",
     ["--no-ortho"], "noortho"),
    ("aux", "no auxiliary deep supervision (w_mid = w_deep = 0)",
     ["--no-aux"], "noaux"),
]:
    t, a = _ab(args_, suffix)
    STUDIES.append(Study(key, "module", label, t, a))

# ---- GROUP 2: bottleneck channel allocation --------------------------------
# The reported model routes 16 of 512 bottleneck channels to the baseline branch.
for bld in (8, 32, 64):
    t, a = _ab(["--base-latent-dim", str(bld)], f"bld{bld}")
    STUDIES.append(Study(f"bld{bld}", "bottleneck",
                         f"baseline/signal split {bld}/{512 - bld}", t, a))

# ---- GROUP 3: encoder width -------------------------------------------------
for w in ((32, 64, 128, 256), (96, 192, 384, 768)):
    t, a = _ab(["--widths"] + [str(x) for x in w], f"w{w[0]}")
    STUDIES.append(Study(f"w{w[0]}", "width",
                         f"encoder widths {'/'.join(map(str, w))}", t, a))

# ---- GROUP 4: comparison (already complete) --------------------------------
STUDIES.append(Study(
    "kazemzadeh", "comparison", "cascaded U-Net (Kazemzadeh et al. 2022)",
    lambda s: f"kazemzadeh_seed{s}",
    lambda s: ["--seed", str(s)],
    script="train_kazemzadeh.py"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", nargs="+", choices=sorted(GROUPS),
                   help="run only these group(s); default is all")
    p.add_argument("--only", nargs="+", metavar="KEY",
                   help="run only these individual studies (keys: %s)"
                        % ", ".join(s.key for s in STUDIES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--epochs", type=int, default=None,
                   help="override the epoch cap, e.g. 2 for a smoke test")
    p.add_argument("--keep-going", action="store_true",
                   help="continue with the remaining studies if one fails")
    args = p.parse_args()

    studies = STUDIES
    if args.group:
        studies = [s for s in studies if s.group in args.group]
    if args.only:
        unknown = set(args.only) - {s.key for s in STUDIES}
        if unknown:
            raise SystemExit(f"unknown study key(s): {', '.join(sorted(unknown))}")
        studies = [s for s in studies if s.key in args.only]

    queue = [(s, seed) for seed in args.seeds for s in studies if not s.done(seed)]
    skipped = [(s, seed) for seed in args.seeds for s in studies if s.done(seed)]

    print("=" * 78)
    print(f"  ABLATION SUITE   {len(queue)} to run, {len(skipped)} already complete")
    print("=" * 78)

    for g in sorted(GROUPS):
        rows = [(s, seed) for (s, seed) in
                [(s, sd) for sd in args.seeds for s in studies] if s.group == g]
        if not rows:
            continue
        pending = [r for r in rows if not r[0].done(r[1])]
        print(f"\n  [{g}]  {GROUPS[g]}")
        print(f"  {len(pending)} to run"
              + (f", ~{len(pending) * HOURS_PER_RUN:.1f} h" if pending else ""))
        for s, seed in rows:
            mark = "done" if s.done(seed) else "  --"
            print(f"    [{mark}] {s.tag(seed):<32} {s.label}")

    if not queue:
        print("\nNothing to do -- every requested study already has results.")
        return

    est = len(queue) * HOURS_PER_RUN
    print("\n" + "-" * 78)
    print(f"  total {len(queue)} runs, estimated {est:.1f} h; expected finish "
          f"{(datetime.now() + timedelta(hours=est)).strftime('%a %d %b %H:%M')}")
    print("  runs are sequential by design: concurrent runs would contend for the GPU")
    print("  and make the per-epoch timings incomparable")

    if args.dry_run:
        print("\n--dry-run: nothing launched.")
        return

    results, t_suite = [], time.perf_counter()
    for i, (s, seed) in enumerate(queue, 1):
        argv = s.argv(seed)
        if args.epochs is not None:
            argv += ["--epochs", str(args.epochs)]
        print("\n" + "=" * 78)
        print(f"  [{i}/{len(queue)}] [{s.group}] {s.label}   (seed {seed})")
        print(f"  {' '.join(argv[1:])}")
        print(f"  started {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 78, flush=True)

        t0 = time.perf_counter()
        rc = subprocess.call(argv, cwd=BASE)
        dt = (time.perf_counter() - t0) / 3600

        ok = (rc == 0 and s.done(seed))
        results.append((s, seed, ok, dt, rc))
        print(f"\n  -> {'OK' if ok else 'FAILED'} (exit {rc}) after {dt:.2f} h")

        if not ok and not args.keep_going:
            print("  stopping; pass --keep-going to continue past a failure.")
            break

    print("\n" + "=" * 78)
    print(f"  SUITE SUMMARY   ({(time.perf_counter() - t_suite) / 3600:.2f} h)")
    print("=" * 78)
    print(f"  {'group':<12}{'study':<30}{'status':>8}{'hours':>8}   metrics")
    for s, seed, ok, dt, rc in results:
        line = f"  {s.group:<12}{s.tag(seed):<30}{'OK' if ok else 'FAIL':>8}{dt:>8.2f}"
        if ok:
            try:
                m = json.load(open(s.meta(seed)))
                line += (f"   MAE {m.get('restored_val_mae', float('nan')):.6f}"
                         f"  cos {m.get('restored_val_cos', float('nan')):.6f}"
                         f"  params {m.get('params_total', 0):,}")
            except Exception:
                pass
        print(line)

    remaining = [(s, sd) for sd in args.seeds for s in STUDIES if not s.done(sd)]
    if remaining:
        print(f"\n  {len(remaining)} study(ies) still outstanding across all groups:")
        for s, sd in remaining:
            print(f"    [{s.group}] {s.tag(sd)}")
    else:
        print("\n  Every study in the suite is complete.")


if __name__ == "__main__":
    main()
