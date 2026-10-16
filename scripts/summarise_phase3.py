"""
Summarise Phase 3: transmission (CT, Eq. 4), H2 raw gain, H3 persistence, partner test.

Reads every <tag>_phase3.json in a folder (written by phase3_eval.py) and reports,
for each condition (mechanism, lambda, rho), the mean over independent runs with a
95% bootstrap confidence interval (10,000 resamples of runs):

  CT        = 1/2 (A_full - A_solo)/E + 1/2 (A_half - A_solo)/E          (Eq. 4)
            A_full, A_half: the student's restraint per encounter with the channel
            on throughout / switched off halfway; A_solo: restraint of the ASOCIAL
            students at the same lambda and rho; E: the teacher's own restraint,
            measured in the same evaluations (mean over all runs at that lambda, rho).
            The bootstrap resamples social and asocial runs together, so the CI
            includes the uncertainty in A_solo. A CI that includes 0 is reported as
            no evidence of transmission (proposal Section 6.7).
  gain      = A_full - A_solo  (H2: compared across teacher levels rho)
  off       restraint with the channel off, before further learning
  retained  = A_post - A_solo_post  (H3): restraint after the post-removal phase,
            relative to asocial students that went through the same phase
  partners  restraint and collective return with the teacher, a greedy agent and
            a naive learner (channel off, learning off for the student)

Everything is computed for TWO restraint measures:
  per encounter  proposal Eq. 10 (an encounter starts each time the student becomes
                 adjacent to a last apple); inflated by back-and-forth movement
  per apple      each last apple counted once, however often the student returns
                 to it (added after the pilot, E14); robust to that movement
Restraint is always printed next to last apples taken per episode and collective
return (proposal Section 6.6), because a higher rate is not protection of the
resource unless fewer last apples are taken.

Usage:
    python scripts/summarise_phase3.py results/pilot
Writes <folder>/phase3_runs.csv (one row per run) and <folder>/phase3_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10_000
MEASURES = ("full", "half", "off", "post", "partner_teacher", "partner_greedy", "partner_naive")


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(0)
    boot = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def joint_ci(social, asocial, stat):
    """Bootstrap a statistic that depends on social AND asocial runs, resampling both."""
    s, a = np.asarray(social, dtype=float), np.asarray(asocial, dtype=float)
    if len(s) == 0 or len(a) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(0)
    point = stat(s, a)
    boots = [stat(rng.choice(s, len(s)), rng.choice(a, len(a))) for _ in range(N_BOOT)]
    return float(point), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def fmt(t, d=3):
    m, lo, hi = t
    if math.isnan(m):
        return "n/a"
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    args = ap.parse_args()
    folder = Path(args.folder)
    files = sorted(folder.glob("*_phase3.json"))
    if not files:
        raise SystemExit(f"no *_phase3.json files in {folder}; run scripts/phase3_eval.py first")

    runs = []
    for f in files:
        d = json.loads(f.read_text())
        c, m = d["config"], d["measures"]
        param = c["kappa"] if c["condition"] == "value_shaping" else c["omega"] if c["condition"] == "decision_biasing" else ""
        row = {"tag": d["tag"], "condition": c["condition"], "param": param, "lam": c["lam"],
               "rho": c["rho"], "seed": c["seed"], "removal_episodes": d["phase3"]["removal_episodes"]}
        for k in MEASURES:
            for q in ("student_restraint", "student_restraint_apple", "student_protected_taken_per_ep",
                      "student_return", "partner_return", "collective_return", "partner_restraint",
                      "partner_restraint_apple", "fully_depleted_frac"):
                row[f"{k}_{q}"] = m[k].get(q, math.nan)  # files from before E14 lack *_apple
        row["half_first"] = m["half"].get("student_restraint_first_half", math.nan)
        row["half_second"] = m["half"].get("student_restraint_second_half", math.nan)
        runs.append(row)

    with (folder / "phase3_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
        w.writeheader()
        w.writerows(runs)

    # ---- sanity check: for asocial students full, half and off must be identical ----
    bad = [r["tag"] for r in runs if r["condition"] == "asocial" and not (
        np.isclose(r["full_student_restraint"], r["half_student_restraint"], equal_nan=True)
        and np.isclose(r["full_student_restraint"], r["off_student_restraint"], equal_nan=True))]
    print("check: asocial full = half = off ..... " + ("OK" if not bad else f"MISMATCH in {bad}"))

    asocial = defaultdict(list)
    teacher_E = defaultdict(list)
    for r in runs:
        teacher_E[(r["lam"], r["rho"])].append(r["full_partner_restraint"])
        if r["condition"] == "asocial":
            asocial[(r["lam"], r["rho"])].append(r)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["condition"], r["param"], r["lam"], r["rho"])].append(r)

    summary = []
    order = {"asocial": 0, "value_shaping": 1, "decision_biasing": 2}
    for key in sorted(groups, key=lambda k: (order.get(k[0], 9), str(k[1]), k[3], k[2])):
        cond, param, lam, rho = key
        g = groups[key]
        a = asocial.get((lam, rho), [])
        E = float(np.nanmean(teacher_E[(lam, rho)]))
        name = cond + (f" kappa={param}" if cond == "value_shaping" else f" omega={param}" if cond == "decision_biasing" else "")
        print(f"\n=== {name}   lambda={lam}   rho={rho}   (n = {len(g)} runs; asocial n = {len(a)}; teacher E = {E:.3f})")
        print(f"   {'measure':18s}{'restraint per encounter':>26s}{'restraint per apple':>26s}"
              f"{'last apples/ep':>16s}{'student ret':>13s}{'collective':>12s}")
        for k in MEASURES:
            if k == "partner_teacher":
                continue  # identical to 'off'
            label = {"off": "off (=teacher)", "partner_greedy": "partner: greedy",
                     "partner_naive": "partner: naive"}.get(k, k)
            print(f"   {label:18s}{fmt(ci([r[f'{k}_student_restraint'] for r in g])):>26s}"
                  f"{fmt(ci([r[f'{k}_student_restraint_apple'] for r in g])):>26s}"
                  f"{np.mean([r[f'{k}_student_protected_taken_per_ep'] for r in g]):>16.2f}"
                  f"{np.mean([r[f'{k}_student_return'] for r in g]):>13.2f}"
                  f"{np.mean([r[f'{k}_collective_return'] for r in g]):>12.2f}")
        row = {"condition": cond, "param": param, "lam": lam, "rho": rho, "n": len(g), "E": E}
        if cond != "asocial" and a:
            for metric, suffix, label in (("student_restraint", "", "per encounter"),
                                          ("student_restraint_apple", "_apple", "per apple")):
                full = [r[f"full_{metric}"] for r in g]
                halfv = [r[f"half_{metric}"] for r in g]
                solo = [r[f"full_{metric}"] for r in a]
                solo_post = [r[f"post_{metric}"] for r in a]
                partner_key = "full_" + metric.replace("student_", "partner_")  # the teacher's own rate
                E_m = float(np.nanmean([r[partner_key] for r in runs if (r["lam"], r["rho"]) == (lam, rho)]))
                if any(math.isnan(v) for v in full + halfv + solo):
                    print(f"   [{label}] not available (re-run phase3_eval.py with --force --reuse-post)")
                    continue
                ct = joint_ci(np.c_[full, halfv], solo,
                              lambda s_, b: 0.5 * (s_[:, 0].mean() - b.mean()) / E_m
                              + 0.5 * (s_[:, 1].mean() - b.mean()) / E_m)
                gain = joint_ci(full, solo, lambda s_, b: s_.mean() - b.mean())
                kept = joint_ci([r[f"post_{metric}"] for r in g], solo_post, lambda s_, b: s_.mean() - b.mean())
                flag = "   <- CI includes 0: no evidence of transmission" if ct[1] <= 0 <= ct[2] else ""
                print(f"   [{label:13s}] CT (Eq. 4) {fmt(ct)}{flag}")
                print(f"   [{label:13s}] gain A_full - A_solo (H2) {fmt(gain)}")
                print(f"   [{label:13s}] retained after removal, A_post - A_solo_post (H3) {fmt(kept)}")
                row.update({f"CT{suffix}": ct[0], f"CT{suffix}_lo": ct[1], f"CT{suffix}_hi": ct[2],
                            f"gain{suffix}": gain[0], f"gain{suffix}_lo": gain[1], f"gain{suffix}_hi": gain[2],
                            f"retained{suffix}": kept[0], f"retained{suffix}_lo": kept[1],
                            f"retained{suffix}_hi": kept[2], f"E{suffix}": E_m})
        for k in MEASURES:
            row[f"{k}_restraint"] = float(np.nanmean([r[f"{k}_student_restraint"] for r in g]))
            row[f"{k}_restraint_apple"] = float(np.nanmean([r[f"{k}_student_restraint_apple"] for r in g]))
            row[f"{k}_taken_per_ep"] = float(np.mean([r[f"{k}_student_protected_taken_per_ep"] for r in g]))
            row[f"{k}_collective"] = float(np.mean([r[f"{k}_collective_return"] for r in g]))
        summary.append(row)

    keys = sorted({k for r in summary for k in r}, key=lambda k: list(summary[0].keys()).index(k)
                  if k in summary[0] else 999)
    with (folder / "phase3_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {folder / 'phase3_runs.csv'} and {folder / 'phase3_summary.csv'}")
    print("Note: with 3 runs per condition, bootstrap intervals are coarse (pilot only).")


if __name__ == "__main__":
    main()
