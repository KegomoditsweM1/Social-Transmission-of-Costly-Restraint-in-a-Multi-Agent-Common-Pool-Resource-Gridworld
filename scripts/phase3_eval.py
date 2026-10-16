"""
Phase 3: evaluate every trained student for transmission, persistence and partners.

For each saved student (<tag>_summary.json + <tag>.npz) this script measures,
with learning OFF unless stated, on the same evaluation seeds as training
(3_000_000 + i) and with the same 5% random moves away from restraint decisions:

  full      social channel ON for the whole episode          -> A_full (Eq. 4)
  half      channel ON, switched OFF halfway through           -> A_half (Eq. 4)
  off       channel OFF for the whole episode (no learning)    -> what the student
            does without the teacher's information, before any further learning;
            informative mainly for decision biasing, whose effect acts at choice time
  post      POST-REMOVAL PHASE (H3, proposal Section 6.5): the student CONTINUES TO
            LEARN for --removal-episodes episodes with its channel OFF (teacher still
            present), then is evaluated with the channel off
  partners  channel OFF, learning off for the student, alongside three partners:
            teacher  its own teacher (same rho); identical to 'off', stored under both names
            greedy   the scripted greedy agent (takes every apple)
            naive    a fresh asocial Q-learner that LEARNS during these episodes

Asocial students go through the same steps (their channel is always off), which
gives A_solo and the baseline for every comparison, including continued learning.

Each measure records: student restraint per encounter (primary), last apples
taken per episode, student / partner / collective return, patch losses by cause,
and the partner's restraint. Output per student: <tag>_phase3.json (and the
post-removal Q-table <tag>_post.npz). Students that already have a _phase3.json
are skipped, so an interrupted run can be restarted.

PROVISIONAL choices (fixed before seeing results; change with the options):
  --removal-episodes 300   length of the post-removal phase (half of training)
  --eval-episodes 50       evaluation episodes per measure
  naive partner            asocial Q-learner with the student's hyper-parameters,
                           learning across the partner-test episodes

Usage (virtual environment active, from the project folder):
    python scripts/phase3_eval.py results/pilot
    python scripts/phase3_eval.py results/pilot --workers 3 --removal-episodes 300
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from agents.q_learner import QLearner

EVAL_SEED_BASE = 3_000_000  # as in train_student.py
T, S = "teacher", "student"  # the partner always occupies the "teacher" slot


# --------------------------------------------------------------------------- #
# pooling episode records (the same formulas as agents.runner.evaluate)
# --------------------------------------------------------------------------- #
def pool(records: list[dict]) -> dict:
    n = len(records)
    out = {"episodes": n,
           "fully_depleted_frac": float(np.mean([r["survival_time"] is not None for r in records])),
           "patch_deaths_simultaneous_per_ep": sum(r["patch_deaths_simultaneous"] for r in records) / n}
    for a, name in ((S, "student"), (T, "partner")):
        took = sum(r[f"{a}_protected_harvests"] for r in records)
        enc = sum(r[f"{a}_encounters"] - r[f"{a}_preempted_encounters"] for r in records)
        out[f"{name}_restraint"] = (1.0 - took / enc) if enc > 0 else math.nan
        out[f"{name}_encounters"] = int(enc)
        out[f"{name}_protected_taken_per_ep"] = took / n
        out[f"{name}_return"] = float(np.mean([r[f"{a}_return"] for r in records]))
        out[f"{name}_patch_deaths_per_ep"] = sum(r[f"{a}_patch_deaths_lapse"] for r in records) / n
    out["collective_return"] = out["student_return"] + out["partner_return"]
    if "student_protected_harvests_first" in records[0]:  # A_half: restraint in each half
        took1 = sum(r["student_protected_harvests_first"] for r in records)
        enc1 = sum(r["student_encounters_first"] - r["student_preempted_encounters_first"] for r in records)
        enc = out["student_encounters"]
        took = out["student_protected_taken_per_ep"] * n
        out["student_restraint_first_half"] = (1.0 - took1 / enc1) if enc1 > 0 else math.nan
        out["student_restraint_second_half"] = (1.0 - (took - took1) / (enc - enc1)) if enc > enc1 else math.nan
    return out


class GreedyActing(QLearner):
    """A frozen student that always acts on its values (argmax with the 5% random
    moves of evaluation), even when run_episode is called with learning and
    exploration on for ANOTHER agent (the naive partner). Must be a QLearner
    subclass, because run_episode recognises learners by their class."""

    def act(self, s, greedy=False, mask=None):
        return super().act(s, greedy=True, mask=mask)


def greedy_acting(q: QLearner) -> QLearner:
    q.__class__ = GreedyActing
    return q


# --------------------------------------------------------------------------- #
# one student
# --------------------------------------------------------------------------- #
def evaluate_student(summary_path: str, removal_episodes: int, eval_episodes: int) -> dict:
    from agents.features import N_STATES
    from agents.runner import run_episode
    from agents.social import DecisionBiasing
    from commons_env import CommonsHarvestEnv
    from commons_env.scripted import GreedyPolicy, TeacherPolicy

    t0 = time.time()
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text())
    cfg, tag = summary["config"], summary["tag"]
    seed, cond = cfg["seed"], cfg["condition"]
    social = cond != "asocial"
    eps = cfg.get("eval_epsilon", 0.05)
    seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + eval_episodes))

    env = CommonsHarvestEnv(lam=cfg["lam"], agent_ids=(T, S), view_radius=cfg.get("view_radius", 4))
    half = env.max_steps // 2

    def student(frozen=True):
        """A fresh copy of the trained student, set up as in train_student.py."""
        q = QLearner.load(summary_path.with_name(tag + ".npz"), seed=seed)
        q.alpha, q.gamma, q.beta = cfg["alpha"], cfg["gamma"], cfg["beta"]
        q.mask_walls = cfg.get("wall_mask", True)
        q.social = None  # value shaping cannot act: the channel is off, or learning is off
        q.bias = DecisionBiasing(cfg["omega"], seed=20_000 + seed) if cond == "decision_biasing" else None
        if frozen:
            q.freeze()
        return q

    def teacher():
        # The SAME teacher seed for every measure (common random numbers): the
        # measures then differ only in what is deliberately changed (channel,
        # removal, partner). For an asocial student, full, half and off are
        # therefore identical, which is checked by summarise_phase3.py.
        return TeacherPolicy(cfg["rho"], seed=10_000 + seed)

    def run(ctrl, channel_on, switch=None, learn=False, ep_seeds=seeds, greedy=True):
        env.set_social_channel(T, False)  # partners never receive social information
        return [run_episode(env, ctrl, seed=s, learn=learn, greedy=greedy,
                            social={S: channel_on}, switch_off_at=switch, eval_epsilon=eps)
                for s in ep_seeds]

    res = {}
    res["full"] = pool(run({T: teacher(), S: student()}, channel_on=social))
    res["half"] = pool(run({T: teacher(), S: student()}, channel_on=social, switch=half))
    res["off"] = pool(run({T: teacher(), S: student()}, channel_on=False))
    res["partner_teacher"] = res["off"]  # the partner test with its own teacher IS the 'off' measure

    # ---- post-removal phase (H3): continue learning with the channel off ----
    learner = student(frozen=False)
    first = cfg["episodes"] + 1  # continue the training seed sequence
    train_seeds = [seed * 100_000 + ep for ep in range(first, first + removal_episodes)]
    ctrl = {T: teacher(), S: learner}
    for s in train_seeds:
        run_episode(env, ctrl, seed=s, learn=True, social={S: False})
    learner.freeze()
    learner.save(summary_path.with_name(tag + "_post.npz"))
    res["post"] = pool(run({T: teacher(), S: learner}, channel_on=False))

    # ---- partner test: channel off, student frozen ----
    res["partner_greedy"] = pool(run({T: GreedyPolicy(seed=50_000 + seed), S: student()}, channel_on=False))
    naive = QLearner(N_STATES, alpha=cfg["alpha"], gamma=cfg["gamma"], beta=cfg["beta"],
                     q_init=cfg.get("q_init", 1.0), seed=40_000 + seed)
    naive.social, naive.bias, naive.mask_walls = None, None, cfg.get("wall_mask", True)
    res["partner_naive"] = pool(run({T: naive, S: greedy_acting(student())}, channel_on=False,
                                    learn=True, greedy=False))

    out = {"tag": tag, "config": cfg,
           "phase3": {"removal_episodes": removal_episodes, "eval_episodes": eval_episodes,
                      "eval_epsilon": eps, "switch_off_at": half,
                      "seconds": round(time.time() - t0)},
           "measures": res}
    summary_path.with_name(tag + "_phase3.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with trained students, e.g. results/pilot")
    ap.add_argument("--removal-episodes", type=int, default=300,
                    help="post-removal learning episodes with the channel off (PROVISIONAL)")
    ap.add_argument("--eval-episodes", type=int, default=50, help="evaluation episodes per measure")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="redo students that already have _phase3.json")
    args = ap.parse_args()

    folder = Path(args.folder)
    todo = []
    for p in sorted(folder.glob("*_summary.json")):
        done = p.with_name(p.name.replace("_summary.json", "_phase3.json"))
        if done.exists() and not args.force:
            print(f"skip (already done): {p.name.replace('_summary.json', '')}")
            continue
        todo.append(str(p))
    print(f"{len(todo)} students to evaluate, {args.workers} at a time "
          f"(removal phase {args.removal_episodes} episodes, {args.eval_episodes} evaluation episodes per measure)")

    t0, ok, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool_:
        futs = {pool_.submit(evaluate_student, p, args.removal_episodes, args.eval_episodes): p for p in todo}
        for i, f in enumerate(as_completed(futs), 1):
            name = Path(futs[f]).name.replace("_summary.json", "")
            try:
                r = f.result()
                m = r["measures"]
                ok += 1
                print(f"[{i}/{len(todo)}] {name}: restraint full {m['full']['student_restraint']:.2f}  "
                      f"off {m['off']['student_restraint']:.2f}  post {m['post']['student_restraint']:.2f}  "
                      f"({r['phase3']['seconds'] / 60:.1f} min, elapsed {(time.time() - t0) / 60:.1f} min)",
                      flush=True)
            except Exception:
                failed += 1
                print(f"[{i}/{len(todo)}] {name}: FAILED\n{traceback.format_exc()}", flush=True)
    print(f"done: {ok} ok, {failed} failed")


if __name__ == "__main__":
    main()
