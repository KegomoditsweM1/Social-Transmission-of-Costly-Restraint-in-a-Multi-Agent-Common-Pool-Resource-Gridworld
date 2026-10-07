"""
Phase 2: train a student beside a frozen (scripted) teacher.

Conditions (proposal Section 6.3):
    asocial   standard Q-learning; the student's social channel is OFF, so it
              receives no information about the teacher's position or actions.
              This is the control condition; its restraint rate is A_solo.
    value_shaping
              social channel ON; whenever the teacher is observed acting, kappa
              is added to Q(s_e, a_e) (Eq. 6; see agents/social.py).
    decision_biasing
              social channel ON; with probability omega the student moves
              towards the teacher's last observed position (Eq. 7), otherwise
              it acts on its own Q-values. Applied in training AND evaluation.

The teacher takes its restrained action with probability rho, otherwise a
greedy one (rho = 1: never takes a last apple). It never learns.

Training and evaluation are separate:
    * training episodes use seeds  seed * 100000 + episode
    * evaluation episodes use seeds 3_000_000 + i, the SAME for every run and
      condition, with learning off: the student acts on its learned values at
      every restraint opportunity, and elsewhere takes a random move with
      probability --eval-epsilon (0.05), so that a deterministic policy cannot
      loop indefinitely.

Usage:
    python scripts/train_student.py --lam 1.0 --seed 0
    python scripts/train_student.py --lam 0.0 --seed 1 --episodes 800

Outputs (in --out-dir, default results/students):
    <tag>.npz            the student's frozen Q-table
    <tag>_curve.csv      learning curve (evaluation checkpoints)
    <tag>_summary.json   configuration and final evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from agents.features import N_STATES
from agents.q_learner import QLearner
from agents.runner import evaluate, run_episode, student_measures
from agents.social import DecisionBiasing, ValueShaping
from commons_env import CommonsHarvestEnv
from commons_env.scripted import TeacherPolicy

EVAL_SEED_BASE = 3_000_000
T, S = "teacher", "student"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="asocial",
                    choices=["asocial", "value_shaping", "decision_biasing"])
    ap.add_argument("--omega", type=float, default=0.2, help="decision-biasing weight (PROVISIONAL)")
    ap.add_argument("--kappa", type=float, default=1.0, help="value-shaping bonus (PROVISIONAL)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1.0, help="teacher restraint level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--beta", type=float, default=5.0)
    ap.add_argument("--q-init", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--final-eval-episodes", type=int, default=50)
    ap.add_argument("--out-dir", default="results/students")
    ap.add_argument("--eval-epsilon", type=float, default=0.05,
                    help="evaluation: random move with this probability, never at a restraint decision")
    ap.add_argument("--view-radius", type=int, default=4,
                    help="how far agents see apples (9 = the whole 10x10 grid); social_radius stays 2")
    ap.add_argument("--no-wall-mask", dest="wall_mask", action="store_false",
                    help="allow moves into the grid edge (the behaviour before E11)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.condition}_lam{args.lam}_rho{args.rho}_s{args.seed}"
    if args.condition == "value_shaping":
        tag = f"{args.condition}_k{args.kappa}_lam{args.lam}_rho{args.rho}_s{args.seed}"
    elif args.condition == "decision_biasing":
        tag = f"{args.condition}_w{args.omega}_lam{args.lam}_rho{args.rho}_s{args.seed}"

    env = CommonsHarvestEnv(lam=args.lam, agent_ids=(T, S), view_radius=args.view_radius)
    teacher = TeacherPolicy(args.rho, seed=10_000 + args.seed)
    student = QLearner(N_STATES, alpha=args.alpha, gamma=args.gamma, beta=args.beta,
                       q_init=args.q_init, seed=args.seed)
    student.social, student.bias = None, None
    student.mask_walls = args.wall_mask  # E11: moves into the grid edge are not available
    if args.condition == "asocial":
        env.set_social_channel(S, False)  # no information about the teacher
    elif args.condition == "value_shaping":
        env.set_social_channel(S, True)
        student.social = ValueShaping(args.kappa)
    else:
        env.set_social_channel(S, True)
        student.bias = DecisionBiasing(args.omega, seed=20_000 + args.seed)
    ctrl = {T: teacher, S: student}

    def checkpoint(ep, n_eval):
        # evaluation must not count towards the TRAINING social-action total
        saved = student.bias.social_actions if student.bias is not None else None
        r = evaluate(env, ctrl, range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval),
                     eval_epsilon=args.eval_epsilon)
        if saved is not None:
            student.bias.social_actions = saved
        return {"episode": ep, **student_measures(r, S, T)}

    rows, t0 = [], time.time()
    for ep in range(1, args.episodes + 1):
        run_episode(env, ctrl, seed=args.seed * 100_000 + ep, learn=True)
        if ep % args.eval_every == 0 or ep == 1:
            rows.append(checkpoint(ep, args.eval_episodes))
            c = rows[-1]
            print(f"[{tag}] ep {ep:4d}  student return {c['student_return']:6.2f}  "
                  f"restraint {c['student_restraint']:.2f}  teacher return {c['teacher_return']:6.2f}  "
                  f"collective {c['collective_return']:6.2f}  ({time.time()-t0:.0f}s)", flush=True)

    with (out / f"{tag}_curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    student.freeze()
    student.save(out / f"{tag}.npz")
    final = checkpoint(args.episodes, args.final_eval_episodes)
    bonuses = student.social.applied if student.social is not None else 0
    social_actions = student.bias.social_actions if student.bias is not None else 0
    summary = {"tag": tag, "config": vars(args), "final": final,
               "social_bonuses_applied": bonuses,
               "social_actions_taken": social_actions,
               "train_seconds": round(time.time() - t0)}
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] FINAL ({args.final_eval_episodes} eval episodes): "
          f"student return {final['student_return']:.2f}  restraint {final['student_restraint']:.3f}  "
          f"collective {final['collective_return']:.2f}")


if __name__ == "__main__":
    main()
