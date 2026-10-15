"""
Stillness check: is a student's standing still copied from the teacher, or its own?

The pilot (E10) found that value-shaping students stand still for long spells.
The scripted teacher pi_R also stands still, by design, whenever no harvestable
apple exists. This script replays the frozen evaluation of every saved student
(learning off, the same evaluation seeds and settings as train_student.py) and
measures, for the TEACHER and the STUDENT side by side:

  still              fraction of steps on which the agent did not move
  still_food_in_view fraction of steps on which it did not move although a
                     harvestable (non-protected) apple was within its view
                     radius: standing still when there was something to eat
  still_beside       fraction of steps standing still next to a protected apple
  beside             fraction of steps next to a protected apple
  stay_chosen        fraction of steps on which it chose the STAY action
  longest_still      mean (over episodes) of the longest spell without moving
  longest_beside     mean of the longest spell standing still next to a
                     protected apple

and, for the world:

  nothing_to_harvest fraction of steps on which no harvestable apple existed
                     anywhere (every apple left was protected, or none at all)

Reading it: if the student's stillness matches the teacher's and both mostly
wait when nothing is harvestable, waiting is what restraint looks like in this
world. If the student often stands still with food in view (and the teacher
does not), the bonus has made it passive.

Because the teacher's random stream is not saved, numbers can differ slightly
from the run's own final evaluation; restraint is re-measured as a check.

Usage (virtual environment active, from the project folder):
    python scripts/stillness_check.py results/pilot
    python scripts/stillness_check.py results/pilot --episodes 20 --workers 3

Writes <dir>/stillness_check.csv and prints a summary grouped by condition and lambda.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

EVAL_SEED_BASE = 3_000_000  # as in train_student.py
T, S = "teacher", "student"
WHO = (T, S)
FIELDS = ("still", "still_food_in_view", "still_beside", "beside", "stay_chosen",
          "longest_still", "longest_beside")


def food_in_view(env, agent, harvestable) -> bool:
    """A harvestable apple within the agent's view radius (square window)."""
    y, x = env.agent_position(agent)
    r = env.view_radius
    return bool(harvestable[max(y - r, 0): y + r + 1, max(x - r, 0): x + r + 1].any())


def check_run(summary_path: str, episodes: int) -> dict:
    from agents.features import N_STATES  # noqa: F401  (imported for clarity of dependency)
    from agents.q_learner import QLearner
    from agents.runner import evaluate
    from agents.social import DecisionBiasing
    from commons_env import CommonsHarvestEnv
    from commons_env.env import STAY
    from commons_env.scripted import TeacherPolicy, adjacent_protected_moves

    summary = json.loads(Path(summary_path).read_text())
    cfg = summary["config"]
    q_path = Path(summary_path).with_name(summary["tag"] + ".npz")

    env = CommonsHarvestEnv(lam=cfg["lam"], agent_ids=(T, S), view_radius=cfg.get("view_radius", 4))
    teacher = TeacherPolicy(cfg["rho"], seed=10_000 + cfg["seed"])
    student = QLearner.load(q_path, seed=cfg["seed"])
    student.freeze()
    student.social, student.bias = None, None  # learning is off, so value shaping has no effect
    student.mask_walls = cfg.get("wall_mask", True)
    if cfg["condition"] == "decision_biasing":  # decision biasing also acts during evaluation
        student.bias = DecisionBiasing(cfg["omega"], seed=20_000 + cfg["seed"])
    env.set_social_channel(S, cfg["condition"] != "asocial")

    # ---- count, step by step, by wrapping env.step on this instance ----
    c = {a: defaultdict(int) for a in WHO}
    world = defaultdict(int)
    spell = {a: 0 for a in WHO}
    longest = {a: [] for a in WHO}  # per-episode longest still spell beside a protected apple
    best = {a: 0 for a in WHO}
    original_step = env.step

    def step(actions):
        if env.episode_stats["steps"] == 0:  # first step of an episode
            for a in WHO:
                if world["steps"] > 0:
                    longest[a].append(best[a])
                spell[a] = best[a] = 0
        apples, prot = env.apple_grid(), env.protected_apple_grid()
        harvestable = apples & ~prot
        world["steps"] += 1
        world["nothing_to_harvest"] += int(not harvestable.any())
        before = {a: env.agent_position(a) for a in WHO}
        beside = {a: bool(adjacent_protected_moves(env, a)) for a in WHO}
        food = {a: food_in_view(env, a, harvestable) for a in WHO}
        out = original_step(actions)
        for a in WHO:
            still = env.agent_position(a) == before[a]
            c[a]["steps"] += 1
            c[a]["still"] += still
            c[a]["still_food_in_view"] += still and food[a]
            c[a]["still_beside"] += still and beside[a]
            c[a]["beside"] += beside[a]
            c[a]["stay_chosen"] += actions.get(a) == STAY
            spell[a] = spell[a] + 1 if (still and beside[a]) else 0
            best[a] = max(best[a], spell[a])
        return out

    env.step = step
    seeds = range(EVAL_SEED_BASE, EVAL_SEED_BASE + episodes)
    r = evaluate(env, {T: teacher, S: student}, seeds, eval_epsilon=cfg.get("eval_epsilon", 0.05))
    for a in WHO:
        longest[a].append(best[a])

    row = {"tag": summary["tag"], "condition": cfg["condition"], "lam": cfg["lam"], "rho": cfg["rho"],
           "kappa": cfg.get("kappa") if cfg["condition"] == "value_shaping" else "",
           "omega": cfg.get("omega") if cfg["condition"] == "decision_biasing" else "",
           "seed": cfg["seed"], "episodes": episodes,
           "nothing_to_harvest": world["nothing_to_harvest"] / world["steps"],
           "student_restraint_recheck": r[f"{S}_restraint"],
           "student_restraint_original": summary["final"]["student_restraint"]}
    for a in WHO:
        n = c[a]["steps"]
        for f in ("still", "still_food_in_view", "still_beside", "beside", "stay_chosen"):
            row[f"{a}_{f}"] = c[a][f] / n
        row[f"{a}_longest_still"] = r[f"{a}_longest_still_mean"]
        row[f"{a}_longest_beside"] = float(np.mean(longest[a]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with <tag>_summary.json and <tag>.npz files, e.g. results/pilot")
    ap.add_argument("--episodes", type=int, default=50,
                    help="evaluation episodes per student (50 = the same seeds as the final evaluation)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    folder = Path(args.folder)
    summaries = sorted(str(p) for p in folder.glob("*_summary.json"))
    if not summaries:
        raise SystemExit(f"no *_summary.json files in {folder}")
    print(f"{len(summaries)} students, {args.episodes} evaluation episodes each, {args.workers} at a time")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(check_run, summaries, [args.episodes] * len(summaries)), 1):
            rows.append(row)
            print(f"[{i}/{len(summaries)}] {row['tag']}: student still {row['student_still']:.2f}, "
                  f"teacher still {row['teacher_still']:.2f}")

    out = folder / "stillness_check.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["kappa"], row["omega"], row["lam"], row["rho"])].append(row)

    def fmt(vals, pct=True):
        m = float(np.mean(vals))
        return f"{m:6.1%}" if pct else f"{m:6.1f}"

    for (cond, kappa, omega, lam, rho), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][3])):
        extra = f"  kappa={kappa}" if kappa != "" else (f"  omega={omega}" if omega != "" else "")
        print(f"\n{cond}{extra}  lambda={lam}  rho={rho}  (n = {len(g)} seeds, means)")
        print(f"   nothing to harvest anywhere: {fmt([r['nothing_to_harvest'] for r in g])} of steps")
        print(f"   {'':22s}{'teacher':>9s}{'student':>9s}")
        for f in FIELDS:
            pct = not f.startswith("longest")
            label = f + ("" if pct else " (steps)")
            print(f"   {label:22s}{fmt([r[f'teacher_{f}'] for r in g], pct):>9s}"
                  f"{fmt([r[f'student_{f}'] for r in g], pct):>9s}")
        rc = [r["student_restraint_recheck"] for r in g]
        ro = [r["student_restraint_original"] for r in g]
        rc_txt = "n/a" if any(math.isnan(v) for v in rc) else f"{np.mean(rc):.3f}"
        print(f"   student restraint: recheck {rc_txt}  (original evaluation {np.mean(ro):.3f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
