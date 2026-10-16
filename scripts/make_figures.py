"""
Figures for the study, from a results folder (e.g. results/pilot or results/full).

Reads the files written by train_student.py (<tag>_curve.csv) and phase3_eval.py
(<tag>_phase3.json) and saves PNG figures in <folder>/figures/:

  1_restraint_by_condition.png  restraint per apple and per encounter, by condition and lambda
  2_resource_outcomes.png       restraint next to last apples taken and collective return
  3_persistence.png             restraint before and after the teacher is removed (H3)
  4_learning_curves.png         restraint and return during training
  5_encounter_inflation.png     per-encounter against per-apple restraint, every run
  6_partners.png                restraint beside the teacher, a greedy agent, a naive learner
  7_H1_ct_vs_cost.png           CT against the measured cost of restraint (Delta) per (lambda, rho)
  8_H2_gain_vs_teacher.png      raw gain A_full - A_solo against teacher level (needs >= 2 rho)

Every point is the mean over independent runs with a 95% bootstrap interval;
small dots are the individual runs. Restraint is per apple unless labelled.

Delta (the cost of restraint) comes from the 200-seed validation gate. The values
below are built in; pass --delta-csv with columns lam,rho,delta,lo,hi to use others.

Usage (virtual environment active, from the project folder):
    python scripts/make_figures.py results/pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarise_phase3 import ci, joint_ci  # noqa: E402  (same bootstrap as the tables)

# ----------------------------------------------------------------------------- #
# style (validated categorical palette, slots 1-3; marker shape as a second code)
# ----------------------------------------------------------------------------- #
COND = {  # condition -> (label, colour, marker)
    "asocial": ("asocial", "#1baf7a", "^"),
    "value_shaping": ("value shaping", "#2a78d6", "o"),
    "decision_biasing": ("decision biasing", "#eb6834", "s"),
}
ORDER = ["asocial", "value_shaping", "decision_biasing"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
LAM_LABEL = {0.0: "λ = 0", 0.5: "λ = ½", 1.0: "λ = 1"}
LAM_MARK = {0.0: "o", 0.5: "D", 1.0: "s"}

# Delta [95% CI] from the validation gate (200 seeds, discounted, averaged over steps)
DEFAULT_DELTA = {
    (0.0, 0.5): (0.97, 0.93, 1.01), (0.0, 0.75): (0.95, 0.91, 0.99), (0.0, 1.0): (0.94, 0.90, 0.98),
    (0.5, 0.5): (0.27, 0.24, 0.29), (0.5, 0.75): (0.24, 0.22, 0.27), (0.5, 1.0): (0.18, 0.15, 0.20),
    (1.0, 0.5): (-0.32, -0.34, -0.30), (1.0, 0.75): (-0.37, -0.39, -0.35), (1.0, 1.0): (-0.63, -0.66, -0.60),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 2,
})


def lam_name(lam):
    return LAM_LABEL.get(lam, f"λ = {lam:g}")


def finite(v):
    return [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]


def legend_handles(conds, extra=()):
    h = [Line2D([], [], color=COND[c][1], marker=COND[c][2], linestyle="", markersize=8, label=COND[c][0])
         for c in conds]
    return h + list(extra)


def bottom_legend(fig, handles):
    """One legend row under the figure, so it never covers data."""
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
               handletextpad=0.4, columnspacing=1.6)


def point_ci(ax, x, values, cond, filled=True, dots=True, size=8, jitter=0.0):
    """Mean with 95% bootstrap CI, plus the individual runs as small dots."""
    label, colour, marker = COND[cond]
    v = finite(values)
    if not v:
        return
    m, lo, hi = ci(v)
    if dots:
        xs = x + np.linspace(-jitter, jitter, len(v)) if len(v) > 1 else [x]
        ax.scatter(xs, v, s=14, color=colour, alpha=0.35, linewidths=0, zorder=2)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=colour, elinewidth=2, capsize=0, zorder=3)
    ax.plot(x, m, marker=marker, markersize=size, color=colour if filled else SURFACE,
            markeredgecolor=colour, markeredgewidth=2, linestyle="", zorder=4)


# ----------------------------------------------------------------------------- #
# data
# ----------------------------------------------------------------------------- #
def load_runs(folder: Path) -> list[dict]:
    runs = []
    for f in sorted(folder.glob("*_phase3.json")):
        d = json.loads(f.read_text())
        c = d["config"]
        runs.append({"tag": d["tag"], "condition": c["condition"], "lam": float(c["lam"]),
                     "rho": float(c["rho"]), "seed": c["seed"], "m": d["measures"]})
    return runs


def val(run, measure, key):
    v = run["m"].get(measure, {}).get(key)
    return math.nan if v is None else float(v)


def groups(runs):
    g = defaultdict(list)
    for r in runs:
        g[(r["condition"], r["lam"], r["rho"])].append(r)
    return g


def dodge(conds):
    w = 0.22
    return {c: (i - (len(conds) - 1) / 2) * w for i, c in enumerate(conds)}


def present(runs, key="condition"):
    have = {r[key] for r in runs}
    return [c for c in ORDER if c in have]


# ----------------------------------------------------------------------------- #
# figures
# ----------------------------------------------------------------------------- #
def fig_restraint(runs, lams, rho, out):
    conds = present(runs)
    g, off = groups(runs), dodge(conds)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, title in ((axes[0], "student_restraint_apple", "Restraint per apple"),
                           (axes[1], "student_restraint", "Restraint per encounter")):
        for i, lam in enumerate(lams):
            for c in conds:
                point_ci(ax, i + off[c], [val(r, "full", key) for r in g.get((c, lam, rho), [])], c, jitter=0.05)
        ax.set_xticks(range(len(lams)), [lam_name(l) for l in lams])
        ax.set_title(title)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("restraint (teacher observable)")
    bottom_legend(fig, legend_handles(conds))
    fig.suptitle(f"Restraint by condition (teacher ρ = {rho:g}; mean, 95% CI, dots = runs)", color=INK, y=1.02)
    fig.savefig(out)
    plt.close(fig)


def fig_outcomes(runs, lams, rho, out):
    conds = present(runs)
    g, off = groups(runs), dodge(conds)
    panels = (("student_restraint_apple", "Restraint per apple", (-0.03, 1.03)),
              ("student_protected_taken_per_ep", "Last apples taken per episode", None),
              ("collective_return", "Collective return per episode", None))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title, ylim) in zip(axes, panels):
        for i, lam in enumerate(lams):
            for c in conds:
                point_ci(ax, i + off[c], [val(r, "full", key) for r in g.get((c, lam, rho), [])], c, jitter=0.05)
        ax.set_xticks(range(len(lams)), [lam_name(l) for l in lams])
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
    bottom_legend(fig, legend_handles(conds))
    fig.suptitle("A higher restraint rate only matters if fewer last apples are taken "
                 f"(teacher ρ = {rho:g})", color=INK, y=1.02)
    fig.savefig(out)
    plt.close(fig)


def fig_persistence(runs, lams, rho, out, delta=None):
    conds = present(runs)
    g = groups(runs)
    fig, axes = plt.subplots(1, len(lams), figsize=(4.0 * len(lams), 3.9), sharey=True, squeeze=False)
    axes = axes[0]
    off = dodge(conds)
    for ax, lam in zip(axes, lams):
        for c in conds:
            label, colour, marker = COND[c]
            rs = g.get((c, lam, rho), [])
            xb, xa = 0 + off[c] * 0.6, 1 + off[c] * 0.6
            for r in rs:  # one thin line per run
                ax.plot([xb, xa], [val(r, "full", "student_restraint_apple"), val(r, "post", "student_restraint_apple")],
                        color=colour, alpha=0.3, linewidth=1, zorder=2)
            b = finite([val(r, "full", "student_restraint_apple") for r in rs])
            a = finite([val(r, "post", "student_restraint_apple") for r in rs])
            if b and a:
                ax.plot([xb, xa], [np.mean(b), np.mean(a)], color=colour, linewidth=2.5, zorder=3)
                point_ci(ax, xb, b, c, dots=False)
                point_ci(ax, xa, a, c, dots=False)
        ax.set_xticks([0, 1], ["teacher observed", "after removal"])
        ax.set_xlim(-0.5, 1.5)
        d = (delta or {}).get((lam, rho))  # the cost of restraint with this teacher, from the gate
        ax.set_title(lam_name(lam) + (f"   (Δ = {d[0]:+.2f})".replace("-", "−") if d else ""))
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("restraint per apple")
    bottom_legend(fig, legend_handles(conds))
    fig.suptitle("Does restraint persist when the student keeps learning without the teacher? "
                 f"(300 episodes; ρ = {rho:g}; Δ = cost of restraint)", color=INK, y=1.03)
    fig.savefig(out)
    plt.close(fig)


def load_curves(folder: Path, runs):
    """{(condition, lam, rho): [list of (episodes, restraint, return)]} from <tag>_curve.csv."""
    curves = defaultdict(list)
    for r in runs:
        f = folder / f"{r['tag']}_curve.csv"
        if not f.exists():
            continue
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
        ep = np.array([float(x["episode"]) for x in rows])
        res = np.array([float(x["student_restraint"]) if x["student_restraint"] not in ("", "nan") else np.nan
                        for x in rows])
        ret = np.array([float(x["student_return"]) for x in rows])
        curves[(r["condition"], r["lam"], r["rho"])].append((ep, res, ret))
    return curves


def fig_curves(folder, runs, lams, rho, out):
    curves = load_curves(folder, runs)
    if not curves:
        print("  (no _curve.csv files: learning curves skipped)")
        return False
    conds = present(runs)
    fig, axes = plt.subplots(2, len(lams), figsize=(4.0 * len(lams), 6.2), sharex=True, squeeze=False)
    for j, lam in enumerate(lams):
        for c in conds:
            label, colour, marker = COND[c]
            cs = curves.get((c, lam, rho), [])
            for k, (ylab, idx) in enumerate((("restraint per encounter", 1), ("student return per episode", 2))):
                ax = axes[k][j]
                for cv in cs:
                    ax.plot(cv[0], cv[idx], color=colour, alpha=0.25, linewidth=1)
                if cs and all(len(cv[0]) == len(cs[0][0]) for cv in cs):
                    mean = np.nanmean(np.vstack([cv[idx] for cv in cs]), axis=0)
                    ax.plot(cs[0][0], mean, color=colour, linewidth=2.2, marker=marker, markersize=5)
                if j == 0:
                    ax.set_ylabel(ylab)
        axes[0][j].set_title(lam_name(lam))
        axes[0][j].set_ylim(-0.03, 1.03)
        axes[1][j].set_ylim(bottom=0)
        axes[1][j].set_xlabel("training episode")
    bottom_legend(fig, legend_handles(conds))
    fig.suptitle(f"Learning curves (evaluation checkpoints; thin = runs, thick = mean; ρ = {rho:g})",
                 color=INK, y=1.01)
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_inflation(runs, out):
    conds = present(runs)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--", zorder=1)
    ax.text(0.97, 0.92, "equal", color=INK2, ha="right", fontsize=9, rotation=45)
    for r in runs:
        label, colour, marker = COND[r["condition"]]
        for meas, filled in (("full", True), ("post", False)):
            x, y = val(r, meas, "student_restraint_apple"), val(r, meas, "student_restraint")
            if math.isnan(x) or math.isnan(y):
                continue
            ax.scatter([x], [y], s=46, marker=marker, color=colour if filled else SURFACE,
                       edgecolors=colour, linewidths=1.6, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("restraint per apple (each last apple counted once)")
    ax.set_ylabel("restraint per encounter (each return counted)")
    ax.grid(True, axis="both")
    shapes = [Line2D([], [], color=INK2, marker="o", linestyle="", markersize=7, label="teacher observed"),
              Line2D([], [], color=INK2, marker="o", markerfacecolor=SURFACE, linestyle="", markersize=7,
                     label="after removal")]
    bottom_legend(fig, legend_handles(conds, shapes))
    ax.set_title("Points above the line: rate inflated by returning to the same apple")
    fig.savefig(out)
    plt.close(fig)


def fig_partners(runs, lams, rho, out):
    conds = present(runs)
    g = groups(runs)
    partners = (("off", "own teacher"), ("partner_greedy", "greedy agent"), ("partner_naive", "naive learner"))
    fig, axes = plt.subplots(1, len(lams), figsize=(4.2 * len(lams), 3.8), sharey=True, squeeze=False)
    axes = axes[0]
    off = dodge(conds)
    for ax, lam in zip(axes, lams):
        for i, (meas, _) in enumerate(partners):
            for c in conds:
                point_ci(ax, i + off[c], [val(r, meas, "student_restraint_apple") for r in g.get((c, lam, rho), [])],
                         c, jitter=0.05)
        ax.set_xticks(range(len(partners)), [p[1] for p in partners])
        ax.set_title(lam_name(lam))
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("restraint per apple (student not observing)")
    bottom_legend(fig, legend_handles(conds))
    fig.suptitle(f"Partner test: restraint with different partners (learning off; ρ = {rho:g})", color=INK, y=1.02)
    fig.savefig(out)
    plt.close(fig)


def ct_cells(runs, metric="student_restraint_apple"):
    """CT [CI] per (condition, lam, rho), as in summarise_phase3.py."""
    g = groups(runs)
    out = {}
    for (c, lam, rho), rs in g.items():
        if c == "asocial" or ("asocial", lam, rho) not in g:
            continue
        a = g[("asocial", lam, rho)]
        partner = metric.replace("student_", "partner_")
        E = float(np.nanmean([val(r, "full", partner) for r in runs if (r["lam"], r["rho"]) == (lam, rho)]))
        full = [val(r, "full", metric) for r in rs]
        half = [val(r, "half", metric) for r in rs]
        solo = [val(r, "full", metric) for r in a]
        if any(math.isnan(v) for v in full + half + solo) or not E:
            continue
        ct = joint_ci(np.c_[full, half], solo,
                      lambda s, b: 0.5 * (s[:, 0].mean() - b.mean()) / E + 0.5 * (s[:, 1].mean() - b.mean()) / E)
        gain = joint_ci(full, solo, lambda s, b: s.mean() - b.mean())
        out[(c, lam, rho)] = {"ct": ct, "gain": gain, "E": E}
    return out


def fig_h1(runs, delta, out):
    cells = ct_cells(runs)
    if not cells:
        print("  (no CT values: H1 figure skipped)")
        return False
    conds = [c for c in ORDER if any(k[0] == c for k in cells)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, only_pos in ((axes[0], False), (axes[1], True)):
        ax.axhline(0, color=INK2, linewidth=1)
        for (c, lam, rho), v in sorted(cells.items()):
            if only_pos and lam == 0.0:
                continue
            if (lam, rho) not in delta:
                continue
            d, dlo, dhi = delta[(lam, rho)]
            m, lo, hi = v["ct"]
            label, colour, _ = COND[c]
            ax.errorbar(d, m, xerr=[[d - dlo], [dhi - d]], yerr=[[m - lo], [hi - m]], fmt="none",
                        ecolor=colour, elinewidth=1.6, zorder=2)
            ax.plot(d, m, marker=LAM_MARK.get(lam, "o"), markersize=8, color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="", zorder=3)
            if len({k[2] for k in cells}) > 1:  # tag the teacher level only when there are several
                ax.annotate(f"ρ={rho:g}", (d, m), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
        ax.axvline(0, color=GRID, linewidth=1, zorder=0)
        ax.set_xlabel("cost of restraint Δ = R_d(1) − R_c(1)   (restraint costly →)")
        ax.set_title("all λ" if not only_pos else "λ > 0 only (restraint protects the resource)")
    axes[0].set_ylabel("CT per apple")
    lam_handles = [Line2D([], [], color=INK2, marker=LAM_MARK[l], linestyle="", markersize=7, label=lam_name(l))
                   for l in sorted({k[1] for k in cells})]
    cond_handles = [Line2D([], [], color=COND[c][1], linewidth=3, label=COND[c][0]) for c in conds]
    bottom_legend(fig, cond_handles + lam_handles)
    n_cells = len({(k[1], k[2]) for k in cells})
    fig.suptitle(f"H1: transmission against the cost of restraint ({n_cells} (λ, ρ) cells; "
                 "error bars: 95% CI on both axes)", color=INK, y=1.02)
    fig.savefig(out)
    plt.close(fig)
    return True


def fig_h2(runs, out):
    cells = ct_cells(runs)
    rhos = sorted({k[2] for k in cells})
    if len(rhos) < 2:
        print("  (only one teacher level: H2 figure skipped until the full design)")
        return False
    lams = sorted({k[1] for k in cells})
    conds = [c for c in ORDER if any(k[0] == c for k in cells)]
    fig, axes = plt.subplots(1, len(lams), figsize=(4.2 * len(lams), 3.8), sharey=True, squeeze=False)
    for ax, lam in zip(axes[0], lams):
        for c in conds:
            label, colour, marker = COND[c]
            pts = sorted((v["E"], v["gain"]) for (cc, l, r), v in cells.items() if cc == c and l == lam)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ax.plot(xs, [p[1][0] for p in pts], color=colour, linewidth=1.5, alpha=0.6)
            for x, (m, lo, hi) in pts:
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=colour, elinewidth=1.6)
                ax.plot(x, m, marker=marker, markersize=8, color=colour, linestyle="")
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_title(lam_name(lam))
        ax.set_xlabel("teacher's measured restraint per apple")
    axes[0][0].set_ylabel("gain A_full − A_solo (per apple)")
    bottom_legend(fig, legend_handles(conds))
    fig.suptitle("H2: do more restrained teachers transmit more?", color=INK, y=1.02)
    fig.savefig(out)
    plt.close(fig)
    return True


def read_delta(path):
    if not path:
        return DEFAULT_DELTA
    d = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            d[(float(row["lam"]), float(row["rho"]))] = (float(row["delta"]), float(row["lo"]), float(row["hi"]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="results folder with *_phase3.json (and *_curve.csv)")
    ap.add_argument("--rho", type=float, default=None,
                    help="teacher level for figures 1-6 (default: the highest present)")
    ap.add_argument("--delta-csv", default=None, help="CSV with lam,rho,delta,lo,hi (default: built-in gate values)")
    args = ap.parse_args()

    folder = Path(args.folder)
    runs = load_runs(folder)
    if not runs:
        raise SystemExit(f"no *_phase3.json in {folder}; run scripts/phase3_eval.py first")
    rho = args.rho if args.rho is not None else max(r["rho"] for r in runs)
    lams = sorted({r["lam"] for r in runs})
    out = folder / "figures"
    out.mkdir(exist_ok=True)
    print(f"{len(runs)} runs; lambda {lams}; figures 1-6 for rho = {rho:g}; saving to {out}")

    made = []
    fig_restraint(runs, lams, rho, out / "1_restraint_by_condition.png"); made.append("1_restraint_by_condition.png")
    fig_outcomes(runs, lams, rho, out / "2_resource_outcomes.png"); made.append("2_resource_outcomes.png")
    delta = read_delta(args.delta_csv)
    fig_persistence(runs, lams, rho, out / "3_persistence.png", delta); made.append("3_persistence.png")
    if fig_curves(folder, runs, lams, rho, out / "4_learning_curves.png"):
        made.append("4_learning_curves.png")
    fig_inflation(runs, out / "5_encounter_inflation.png"); made.append("5_encounter_inflation.png")
    fig_partners(runs, lams, rho, out / "6_partners.png"); made.append("6_partners.png")
    if fig_h1(runs, delta, out / "7_H1_ct_vs_cost.png"):
        made.append("7_H1_ct_vs_cost.png")
    if fig_h2(runs, out / "8_H2_gain_vs_teacher.png"):
        made.append("8_H2_gain_vs_teacher.png")
    for m in made:
        print(f"  saved {out / m}")


if __name__ == "__main__":
    main()
