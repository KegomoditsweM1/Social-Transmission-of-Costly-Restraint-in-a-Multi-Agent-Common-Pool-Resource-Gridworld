"""
Choose the value-shaping bonus kappa and the decision-biasing weight omega.

The rule is fixed here BEFORE any selection run, so the choice cannot be bent
towards a desired result:

  * Selection runs use the no-dilemma control (lambda = 0) with the fully
    restrained teacher (rho = 1), on seeds that are NOT reused in the main runs
    (e.g. 100, 101, 102), plus asocial students on the same seeds.
  * For each scanned value, with A the mean restraint per encounter of the
    students, A_solo that of the asocial students and E = 1 the teacher's:

        f = (A - A_solo) / (E - A_solo)

    is the fraction of the gap between asocial student and teacher that the
    mechanism closes.
  * A value gives PARTIAL transmission if 0.25 <= f <= 0.75.
  * The chosen value is the partial value with f closest to 0.5.
  * The sensitivity value is the scanned neighbour of the chosen value (the next
    larger value if there is one, otherwise the next smaller one).
  * If no value is partial, the value whose f lies CLOSEST to the partial range
    [0.25, 0.75] is chosen, and the absence of partial transmission is reported
    as a result (it is not hidden by scanning further).

Usage:
    python scripts/select_social_param.py results/selection
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

LOW, HIGH, TARGET = 0.25, 0.75, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--teacher-restraint", type=float, default=1.0, help="E (the teacher's restraint)")
    args = ap.parse_args()

    groups = defaultdict(list)  # (condition, value) -> [final dicts]
    for f in sorted(Path(args.folder).glob("*_summary.json")):
        d = json.loads(f.read_text())
        c = d["config"]
        if c["lam"] != args.lam or c["rho"] != args.rho:
            continue
        value = c.get("kappa") if c["condition"] == "value_shaping" else \
            c.get("omega") if c["condition"] == "decision_biasing" else None
        groups[(c["condition"], value)].append((c["seed"], d["final"]))

    solo = groups.get(("asocial", None), [])
    if not solo:
        raise SystemExit("no asocial runs found with these settings: they are needed for A_solo")
    a_solo = float(np.nanmean([fi["student_restraint"] for _, fi in solo]))
    E = args.teacher_restraint
    print(f"lambda = {args.lam}, rho = {args.rho}:  A_solo = {a_solo:.3f} "
          f"(asocial, seeds {sorted(s for s, _ in solo)}),  E = {E}")
    print(f"rule: partial if {LOW} <= f <= {HIGH}; choose f closest to {TARGET}\n")

    for cond, pname in (("value_shaping", "kappa"), ("decision_biasing", "omega")):
        values = sorted(v for (c, v) in groups if c == cond)
        if not values:
            continue
        print(f"{cond}  ({pname})")
        rows = []
        for v in values:
            items = groups[(cond, v)]
            A = float(np.nanmean([fi["student_restraint"] for _, fi in items]))
            f = (A - a_solo) / (E - a_solo) if E > a_solo else math.nan
            taken = float(np.mean([fi["student_protected_taken_per_ep"] for _, fi in items]))
            ret = float(np.mean([fi["student_return"] for _, fi in items]))
            partial = LOW <= f <= HIGH
            rows.append((v, f, partial))
            print(f"   {pname} = {v:<7}  restraint {A:.3f}  f = {f:6.3f}  {'PARTIAL' if partial else '       '}  "
                  f"last apples taken/ep {taken:5.2f}  return {ret:6.2f}  "
                  f"(seeds {sorted(s for s, _ in items)})")
        partial_rows = [r for r in rows if r[2]]
        if partial_rows:
            chosen = min(partial_rows, key=lambda r: abs(r[1] - TARGET))[0]
            note = ""
        else:  # fallback fixed in advance: closest to the partial range
            dist = lambda f: (LOW - f) if f < LOW else (f - HIGH) if f > HIGH else 0.0
            chosen = min(rows, key=lambda r: dist(r[1]) if not math.isnan(r[1]) else math.inf)[0]
            note = "  (NO value gave partial transmission: report this; chosen by the fallback rule)"
        i = values.index(chosen)
        sens = values[i + 1] if i + 1 < len(values) else (values[i - 1] if i > 0 else None)
        print(f"   -> chosen {pname} = {chosen};  sensitivity value {pname} = {sens}{note}\n")


if __name__ == "__main__":
    main()
