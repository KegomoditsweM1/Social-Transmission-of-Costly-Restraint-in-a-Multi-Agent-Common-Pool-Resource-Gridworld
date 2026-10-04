"""
Phase 1: train the teacher (proposal Section 6.3).

The teacher is a tabular Q-learner (Eq. 2, softmax) trained ALONE at
lambda = 1, then frozen. Moves off the grid are unavailable (Section 6.2).

Every --eval-every episodes it is evaluated with learning off on separate
evaluation seeds (Section 6.5 rule: learned values at every restraint
opportunity, a random move with probability --eval-epsilon elsewhere), giving
a learning curve of its return and its restraint per encounter (Eq. 10).

Reliability check: the proposal falls back to the scripted restrained policy
pi_R if the trained teacher "does not reliably avoid taking the last apple".
This script reports whether its final restraint per encounter reaches
--reliable (default 0.95, a PROVISIONAL threshold to agree with the supervisor).

Usage:
    python scripts/train_teacher.py --seed 0
    python scripts/train_teacher.py --seed 0 --gamma 0.999 --episodes 1000

Outputs (in --out-dir, default results/teacher):
    teacher_g{gamma}_s{seed}.npz           frozen Q-table
    teacher_g{gamma}_s{seed}_curve.csv     learning curve
    teacher_g{gamma}_s{seed}_summary.json  configuration and final evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from agents.features import N_STATES
from agents.q_learner import QLearner
from agents.runner import evaluate, run_episode
from commons_env import CommonsHarvestEnv

EVAL_SEED_BASE = 1_000_000  # evaluation seeds never overlap training seeds
A = "teacher"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--q-init", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--final-eval-episodes", type=int, default=50)
    ap.add_argument("--eval-epsilon", type=float, default=0.05)
    ap.add_argument("--reliable", type=float, default=0.95,
                    help="restraint per encounter needed to count as 'reliably restrained'")
    ap.add_argument("--out-dir", default="results/teacher")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"teacher_g{args.gamma}_s{args.seed}"

    env = CommonsHarvestEnv(lam=args.lam, agent_ids=(A,))
    q = QLearner(N_STATES, alpha=args.alpha, gamma=args.gamma, beta=args.beta,
                 q_init=args.q_init, seed=args.seed)
    q.mask_walls = True  # moves off the grid are unavailable (Section 6.2)
    ctrl = {A: q}

    def measure(n):
        r = evaluate(env, ctrl, range(EVAL_SEED_BASE, EVAL_SEED_BASE + n), eval_epsilon=args.eval_epsilon)
        return {"return": r[f"{A}_return_mean"],
                "restraint": r[f"{A}_restraint"],            # per encounter (Eq. 10, primary)
                "restraint_step": r[f"{A}_restraint_step"],  # per step (secondary)
                "protected_taken_per_ep": r[f"{A}_protected_taken_per_ep"],
                "encounters": r[f"{A}_encounters_total"],
                "fully_depleted_frac": r["fully_depleted_frac"]}

    rows, t0 = [], time.time()
    for ep in range(1, args.episodes + 1):
        # training seeds: seed * 100000 + episode, disjoint from evaluation seeds
        run_episode(env, ctrl, seed=args.seed * 100_000 + ep, learn=True)
        if ep % args.eval_every == 0 or ep == 1:
            rows.append({"episode": ep, **measure(args.eval_episodes)})
            c = rows[-1]
            print(f"[{tag}] ep {ep:5d}  return {c['return']:6.2f}  restraint {c['restraint']:.3f}  "
                  f"last apples taken/ep {c['protected_taken_per_ep']:.2f}  ({time.time() - t0:.0f}s)",
                  flush=True)

    with (out / f"{tag}_curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    q.freeze()
    q.save(out / f"{tag}.npz")
    final = measure(args.final_eval_episodes)
    reliable = final["restraint"] >= args.reliable
    summary = {"config": vars(args), "final_eval": final, "reliable": reliable,
               "train_seconds": round(time.time() - t0)}
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] FINAL ({args.final_eval_episodes} episodes): return {final['return']:.2f}  "
          f"restraint per encounter {final['restraint']:.3f}  "
          f"-> {'RELIABLY restrained' if reliable else 'NOT reliably restrained'} "
          f"(threshold {args.reliable})")


if __name__ == "__main__":
    main()
