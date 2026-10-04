"""
Phase 0 validation gate (proposal Section 6.2).

For every dilemma strength lambda and every teacher level rho, scripted
policies are paired as in a two-player Schelling diagram, from the STUDENT's
point of view (l = number of OTHER agents that cooperate):

    Rc(1): student restrained (pi_R)  beside the teacher pi_T^rho   (Eq. 9)
    Rd(1): student greedy     (pi_G)  beside the teacher pi_T^rho
    Rc(0): student restrained (pi_R)  beside a greedy partner (pi_G)
    Rd(0): student greedy     (pi_G)  beside a greedy partner (pi_G)

    greed gap  Delta = Rd(1) - Rc(1)   (restraint is individually costly iff Delta > 0)

Each return is computed three ways:
    undiscounted   total reward in the episode
    disc_avg       PRIMARY: return discounted with gamma, computed from EVERY step
                   of the episode and averaged (the student values every state it visits)
    disc_start     return discounted with gamma from the start of the episode

Comparisons are PAIRED by seed (same environment seed and partner seed), and
each holds only if the 95% bootstrap interval of the difference lies above 0.

Social dilemma (Section 5.3): (1) Rc(1) > Rd(0), (2) Rc(1) > Rc(0), and
fear Rd(0) > Rc(0) or greed Rd(1) > Rc(1).

Gate (judged on the PRIMARY measure):
    A  a social dilemma at every lambda > 0, with the fully restrained teacher (rho = 1)
    B  no social dilemma at lambda = 0, at any rho
    C  at least one condition with Delta > 0. Reported separately for lambda > 0,
       because at lambda = 0 restraint protects nothing, so Delta > 0 there is expected.

Usage:
    python scripts/check_gate.py
    python scripts/check_gate.py --seeds 50 --max-steps 1500 --p-bar 0.001 --workers 3
    python scripts/check_gate.py --seeds 5 --max-steps 300          # quick smoke test

Outputs in --out-dir (default results/gate): per-seed CSV, summary JSON, one
Schelling-diagram figure per return measure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MEASURES = ("disc_avg", "undiscounted", "disc_start")
PRIMARY = "disc_avg"


def returns_from_rewards(rew: np.ndarray, gamma: float) -> dict:
    """rew[t] = reward received on step t+1 of the episode (t = 0..T-1).
    G_t = sum_k gamma^k rew[t+k], truncated at the end of the episode."""
    T = len(rew)
    G = np.zeros(T + 1)
    for t in range(T - 1, -1, -1):
        G[t] = rew[t] + gamma * G[t + 1]
    return {"undiscounted": float(rew.sum()), "disc_start": float(G[0]),
            "disc_avg": float(G[:T].mean())}


def run_task(task: dict) -> list[dict]:
    """All seeds of one pairing: (lambda, student policy, partner policy)."""
    from commons_env import CommonsHarvestEnv
    from commons_env.scripted import GreedyPolicy, RestrainedPolicy, TeacherPolicy

    env = CommonsHarvestEnv(lam=task["lam"], p_bar=task["p_bar"], max_steps=task["max_steps"],
                            agent_ids=("partner", "student"))
    rows = []
    for seed in range(task["seeds"]):
        student = (RestrainedPolicy(seed=20_000 + seed) if task["student"] == "C"
                   else GreedyPolicy(seed=20_000 + seed))
        partner = (GreedyPolicy(seed=10_000 + seed) if task["partner"] == "greedy"
                   else TeacherPolicy(task["rho"], seed=10_000 + seed))
        env.reset(seed=seed)
        rew = []
        while env.agents:
            _, r, _, _, _ = env.step({"partner": partner.act(env, "partner"),
                                      "student": student.act(env, "student")})
            rew.append(r["student"])
        rows.append({"lam": task["lam"], "rho": task["rho"], "student": task["student"],
                     "partner": task["partner"], "seed": seed,
                     **returns_from_rewards(np.array(rew, float), task["gamma"])})
    return rows


def paired_ci(a, b, n_boot=5000, seed=0):
    """Mean of (a - b) over paired seeds, with a 95% bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holds(ci) -> str:
    """A comparison holds ('yes') only if its 95% interval lies above 0, fails
    ('no') only if it lies below 0, and is 'unclear' otherwise."""
    return "yes" if ci[1] > 0 else ("no" if ci[2] < 0 else "unclear")


def classify(cond1: str, cond2: str, fear: str, greed: str) -> str:
    """Social dilemma: (1) and (2) must hold, and at least one of fear / greed.
    'yes'     every required condition is established;
    'no'      a required condition is ruled out (its interval lies below 0),
              or both temptations are ruled out;
    'unclear' otherwise (more seeds are needed to decide)."""
    if cond1 == "yes" and cond2 == "yes" and "yes" in (fear, greed):
        return "yes"
    if "no" in (cond1, cond2) or (fear == "no" and greed == "no"):
        return "no"
    return "unclear"


def mean_ci(x, n_boot=5000, seed=0):
    """Mean of x with a 95% bootstrap interval (for error bars)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--p-bar", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=0.99, help="the student's discount factor")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out-dir", default="results/gate")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    common = {"seeds": args.seeds, "max_steps": args.max_steps, "p_bar": args.p_bar, "gamma": args.gamma}
    tasks = []
    for lam in args.lams:
        for st in ("C", "D"):  # l = 0: greedy partner (does not depend on rho)
            tasks.append({**common, "lam": lam, "rho": None, "student": st, "partner": "greedy"})
            for rho in args.rhos:  # l = 1: the teacher at level rho
                tasks.append({**common, "lam": lam, "rho": rho, "student": st, "partner": "teacher"})
    print(f"{len(tasks)} pairings x {args.seeds} seeds, {args.workers} at a time ...", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [r for chunk in pool.map(run_task, tasks) for r in chunk]

    with (out / "gate_per_seed.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def series(lam, rho, student, partner, m):
        sel = [r for r in rows if r["lam"] == lam and r["student"] == student and r["partner"] == partner
               and (partner == "greedy" or r["rho"] == rho)]
        return [r[m] for r in sorted(sel, key=lambda r: r["seed"])]

    summary = {"settings": vars(args), "conditions": []}
    for lam in args.lams:
        for rho in args.rhos:
            cond = {"lam": lam, "rho": rho}
            for m in MEASURES:
                rc1, rd1 = series(lam, rho, "C", "teacher", m), series(lam, rho, "D", "teacher", m)
                rc0, rd0 = series(lam, rho, "C", "greedy", m), series(lam, rho, "D", "greedy", m)
                c1, c2 = paired_ci(rc1, rd0), paired_ci(rc1, rc0)
                fear, greed = paired_ci(rd0, rc0), paired_ci(rd1, rc1)
                dilemma = classify(holds(c1), holds(c2), holds(fear), holds(greed))
                cond[m] = {"Rc1": mean_ci(rc1), "Rd1": mean_ci(rd1),
                           "Rc0": mean_ci(rc0), "Rd0": mean_ci(rd0),
                           "Delta": greed, "cond1": holds(c1), "cond2": holds(c2),
                           "fear": holds(fear), "greed": holds(greed), "dilemma": dilemma}
            summary["conditions"].append(cond)

    for m in MEASURES:
        tag = " (PRIMARY)" if m == PRIMARY else ""
        print(f"\n=== {m}{tag} ===")
        print(f"{'lambda':>6} {'rho':>5} {'Rc(1)':>8} {'Rd(1)':>8} {'Rc(0)':>8} {'Rd(0)':>8}"
              f"  {'Delta [95% CI]':>26}  (1)     (2)     fear    greed   dilemma")
        for c in summary["conditions"]:
            v = c[m]
            d = v["Delta"]
            print(f"{c['lam']:6.2f} {c['rho']:5.2f} {v['Rc1'][0]:8.3f} {v['Rd1'][0]:8.3f} {v['Rc0'][0]:8.3f} "
                  f"{v['Rd0'][0]:8.3f}  {d[0]:8.3f} [{d[1]:7.3f}, {d[2]:7.3f}]  {v['cond1']:7s} {v['cond2']:7s} "
                  f"{v['fear']:7s} {v['greed']:7s} {v['dilemma'].upper() if v['dilemma'] == 'yes' else v['dilemma']}")

    P = {(c["lam"], c["rho"]): c[PRIMARY] for c in summary["conditions"]}
    pos = [lam for lam in args.lams if lam > 0]
    A = all(P[(lam, 1.0)]["dilemma"] == "yes" for lam in pos) if 1.0 in args.rhos else None
    B = all(P[(0.0, rho)]["dilemma"] == "no" for rho in args.rhos) if 0.0 in args.lams else None
    unclear = [(lam, rho) for (lam, rho), v in P.items() if v["dilemma"] == "unclear"]
    C_any = any(v["Delta"][1] > 0 for v in P.values())
    C_pos = any(v["Delta"][1] > 0 for (lam, _), v in P.items() if lam > 0)
    summary["gate"] = {"A_dilemma_at_lambda_gt_0_rho_1": A, "B_no_dilemma_at_lambda_0": B,
                       "C_Delta_gt_0_any_condition": C_any, "C_Delta_gt_0_some_lambda_gt_0": C_pos,
                       "passes": bool(A and B and C_any)}
    print(f"\nGATE (on {PRIMARY}):")
    if unclear:
        print(f"   note: dilemma status UNCLEAR at (lambda, rho) = {unclear}: more seeds are needed there")
    print(f"   A  social dilemma at every lambda > 0 with rho = 1 : {A}")
    print(f"   B  no social dilemma at lambda = 0 (any rho)      : {B}")
    print(f"   C  some condition with Delta > 0                  : {C_any}")
    print(f"      ... and some condition with lambda > 0         : {C_pos}   (the informative case)")
    subset = (0.0 not in args.lams or not pos or 1.0 not in args.rhos)
    summary["gate"]["subset_run"] = subset
    if subset:
        print("   -> NO VERDICT: this run covers only part of the conditions (the gate needs "
              "lambda = 0, at least one lambda > 0, and rho = 1). Combine it with a full run.")
    else:
        print(f"   -> gate {'PASSES' if summary['gate']['passes'] else 'FAILS'}"
              f"{'' if C_pos else '  (note: Delta > 0 only where restraint protects nothing)'}")
    (out / "gate_summary.json").write_text(json.dumps(summary, indent=2))

    plot(summary, args, out)
    print(f"\nwritten {out / 'gate_per_seed.csv'}, {out / 'gate_summary.json'}, figures in {out}")


def plot(summary, args, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for m in MEASURES:
        fig, axes = plt.subplots(len(args.lams), len(args.rhos), squeeze=False,
                                 figsize=(3.8 * len(args.rhos), 3.2 * len(args.lams)))
        for c in summary["conditions"]:
            i, j = args.lams.index(c["lam"]), args.rhos.index(c["rho"])
            v, ax = c[m], axes[i][j]
            for key0, key1, colour, marker, label in (("Rc0", "Rc1", "#2d69be", "o", "restrained $R_c$"),
                                                      ("Rd0", "Rd1", "#d9534f", "s", "greedy $R_d$")):
                means = [v[key0][0], v[key1][0]]
                err = [[means[0] - v[key0][1], means[1] - v[key1][1]],
                       [v[key0][2] - means[0], v[key1][2] - means[1]]]
                ax.errorbar([0, 1], means, yerr=err, color=colour, marker=marker, lw=2,
                            capsize=4, label=label)
            d = v["Delta"]
            ax.set_title(f"$\\lambda$={c['lam']:g}, $\\rho$={c['rho']:g}   dilemma: {v['dilemma']}\n"
                         f"$\\Delta$ = {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]", fontsize=8.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["greedy\npartner", "teacher\n$\\rho$"], fontsize=8)
            ax.grid(alpha=0.3)
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"Schelling diagrams (student's return: {m}"
                     f"{', PRIMARY' if m == PRIMARY else ''}; gamma = {args.gamma}; "
                     f"{args.seeds} seeds; error bars: 95% CI)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"schelling_{m}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
