"""
Phase 1: create the three teacher levels (proposal Section 6.3, Eq. 9) and
MEASURE their restraint (Eq. 10).

    pi_T^rho = rho * pi_R + (1 - rho) * pi_G,   rho in {0.5, 0.75, 1}

i.e. the teacher behaves as its restrained policy, except that whenever a last
apple is adjacent it takes it with probability 1 - rho. The nominal rho is not
the realised rate, so the MEASURED restraint is reported and used as the
teacher's level.

The restrained policy is either
  * the scripted pi_R                            (--teacher scripted), or
  * a trained, frozen Q-learner from Phase 1     (--teacher results/teacher/<file>.npz).

Measurement: the teacher alone at --lam (default 1), on evaluation seeds disjoint
from training. Restraint is pooled over episodes; its 95% CI is a bootstrap
over episodes.
    restraint       PRIMARY: per encounter (Eq. 10)
    restraint_step  secondary: per step adjacent to a last apple

Usage:
    python scripts/make_teacher_levels.py --teacher scripted
    python scripts/make_teacher_levels.py --teacher results/teacher/teacher_g0.99_s0.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from agents.q_learner import QLearner
from agents.runner import FrozenTeacher, run_episode
from commons_env import CommonsHarvestEnv
from commons_env.scripted import TeacherPolicy

EVAL_SEED_BASE = 2_000_000
A = "teacher"


def pooled(recs, per="encounter"):
    """Pooled restraint over a list of episode records."""
    took = sum(r[f"{A}_protected_harvests"] for r in recs)
    if per == "encounter":
        opps = sum(r[f"{A}_encounters"] - r[f"{A}_preempted_encounters"] for r in recs)
    else:
        opps = sum(r[f"{A}_opportunities"] - r[f"{A}_preempted"] for r in recs)
    return (1 - took / opps) if opps > 0 else math.nan


def bootstrap(recs, per="encounter", n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array([pooled([recs[i] for i in rng.integers(len(recs), size=len(recs))], per)
                     for _ in range(n_boot)])
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="'scripted' or path to a frozen Q-table (.npz)")
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", default="results/teacher/teacher_levels.json")
    args = ap.parse_args()

    env = CommonsHarvestEnv(lam=args.lam, agent_ids=(A,))
    seeds = range(EVAL_SEED_BASE, EVAL_SEED_BASE + args.episodes)
    results = {"teacher": args.teacher, "lam": args.lam, "episodes": args.episodes, "levels": []}

    for rho in args.rhos:
        if args.teacher == "scripted":
            ctrl = TeacherPolicy(rho, seed=int(rho * 1000))
        else:
            q = QLearner.load(args.teacher, seed=0)
            q.mask_walls = True
            ctrl = FrozenTeacher(q, rho=rho, seed=int(rho * 1000))
        recs = [run_episode(env, {A: ctrl}, seed=s, learn=False, greedy=True) for s in seeds]
        res, res_step = pooled(recs), pooled(recs, "step")
        lo, hi = bootstrap(recs)
        level = {
            "rho": rho,
            "restraint": res, "restraint_ci": [lo, hi],
            "restraint_step": res_step,
            "protected_taken_per_ep": float(np.mean([r[f"{A}_protected_harvests"] for r in recs])),
            "encounters_total": int(sum(r[f"{A}_encounters"] for r in recs)),
            "return_mean": float(np.mean([r[f"{A}_return"] for r in recs])),
            "fully_depleted_frac": float(np.mean([r["survival_time"] is not None for r in recs])),
        }
        results["levels"].append(level)
        print(f"rho = {rho:4.2f}: restraint per encounter {res:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"(per step {res_step:.3f})  return {level['return_mean']:.2f}  "
              f"last apples taken/ep {level['protected_taken_per_ep']:.2f}  "
              f"all patches dead {100 * level['fully_depleted_frac']:.0f}%")

    ci = [lv["restraint_ci"] for lv in results["levels"]]
    separated = all(ci[i][1] < ci[i + 1][0] for i in range(len(ci) - 1))
    results["levels_separated"] = separated
    print(f"adjacent levels have non-overlapping 95% CIs: {separated}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
