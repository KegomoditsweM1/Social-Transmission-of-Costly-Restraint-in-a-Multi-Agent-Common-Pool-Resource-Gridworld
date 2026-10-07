"""
Summarise student runs across seeds (proposal Section 6.7).

The unit of analysis is the independent training run: each seed contributes
ONE value per measure (from its final frozen evaluation). For each condition
this reports the mean, a 95% bootstrap confidence interval over seeds, and the
range. With few seeds the interval is crude; the range is reported so that
this is visible.

Usage:
    python scripts/summarise_students.py results/students
    python scripts/summarise_students.py results/students results/kappa001
    python scripts/summarise_students.py results/students --out results/students/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

KEY_FIELDS = ["condition", "kappa", "omega", "lam", "rho", "episodes", "alpha", "gamma", "beta",
              "q_init", "eval_every", "eval_episodes", "final_eval_episodes", "wall_mask",
              "eval_epsilon", "view_radius"]
DEFAULTS = {"rho": 1.0, "episodes": 600, "alpha": 0.1, "gamma": 0.99, "beta": 5.0,
            "q_init": 1.0, "eval_every": 100, "eval_episodes": 20, "final_eval_episodes": 50,
            "wall_mask": True, "eval_epsilon": 0.05, "view_radius": 4}  # runs before E11 have wall_mask=None (unmasked)

MEASURES = ["student_restraint", "student_protected_taken_per_ep", "student_restraint_step",
            "student_return", "student_harvests", "student_encounters",
            "student_opportunities", "teacher_return", "collective_return",
            "fully_depleted_frac", "patch_deaths_by_student_per_ep",
            "patch_deaths_by_teacher_per_ep", "patch_deaths_simultaneous_per_ep",
            "student_wall_blocked_frac", "student_agent_blocked_frac",
            "student_longest_still_mean"]


def bootstrap_ci(x, n_boot=5000, seed=0):
    x = np.asarray([v for v in x if not math.isnan(v)], dtype=float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = defaultdict(list)
    files = sorted(f for folder in args.folders for f in Path(folder).glob("*_summary.json"))
    for f in files:
        d = json.loads(f.read_text())
        c = d["config"]
        # EVERY setting that can change a run's result is part of the key;
        # only the seed and output folder may differ within a group.
        c = {**c, "view_radius": c.get("view_radius", 4)}  # runs before E14 used 4
        key = tuple((k, c.get(k)) for k in KEY_FIELDS)
        # a mechanism's parameter only belongs to that mechanism's key
        unused = {"asocial": {"kappa", "omega"}, "value_shaping": {"omega"},
                  "decision_biasing": {"kappa"}}[c["condition"]]
        key = tuple((k, None if k in unused else v) for k, v in key)
        groups[key].append((c["seed"], d["final"]))

    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cfg = dict(key)
        seeds = [sd for sd, _ in items]
        finals = [fi for _, fi in items]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds {seeds} in group {cfg}")
        kap = f"  kappa={cfg['kappa']}" if cfg["kappa"] is not None else ""
        kap += f"  omega={cfg['omega']}" if cfg["omega"] is not None else ""
        # show any setting that differs from the defaults, so no difference is hidden
        extra = "  ".join(f"{k}={v}" for k, v in cfg.items()
                          if k in DEFAULTS and k != "rho" and v != DEFAULTS[k])
        print(f"\n{cfg['condition']}{kap}  lambda={cfg['lam']}  rho={cfg['rho']}"
              f"{'  [' + extra + ']' if extra else ''}  (n = {len(finals)} seeds: {seeds})")
        row = {**cfg, "n_seeds": len(finals)}
        for m in MEASURES:
            if not all(m in fi for fi in finals):
                continue  # measure absent in older result files
            vals = [float(fi[m]) for fi in finals]
            mean, lo, hi = bootstrap_ci(vals)
            finite = [v for v in vals if not math.isnan(v)]
            print(f"   {m:22s} {mean:8.3f}  95% CI [{lo:8.3f}, {hi:8.3f}]   "
                  f"range [{min(finite):.3f}, {max(finite):.3f}]   per seed {[round(v, 3) for v in vals]}")
            row.update({f"{m}_mean": mean, f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(row)

    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
