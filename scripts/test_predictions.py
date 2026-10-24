"""
Test the predictions of PREDICTIONS.md (P1-P6) on a results folder.

Reads <folder>/phase3_runs.csv, which scripts/summarise_phase3.py writes, so run
that first. Primary measure: restraint per apple (pass --metric to use another).

    python scripts/summarise_phase3.py results/full
    python scripts/test_predictions.py results/full

Every interval is a 95% bootstrap interval over runs (10,000 resamples), each
group of runs resampled independently. Verdicts follow PREDICTIONS.md:
  supported     the interval lies entirely on the predicted side
  contradicted  it lies entirely on the other side
  inconclusive  otherwise
Cells missing from the folder (for example teacher levels the pilot did not
run) are reported as "not available". Writes <folder>/predictions.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT = 10_000
LAMS = (0.0, 0.5, 1.0)
RHOS = (0.5, 0.75, 1.0)
CT_MAX_DB = 0.10  # P6: decision biasing transmits "almost nothing"


def load(folder: Path):
    path = folder / "phase3_runs.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found; run scripts/summarise_phase3.py {folder} first")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["condition"], float(r["lam"]), float(r["rho"]))].append(r)
    return groups


def col(rs, name):
    v = np.array([float(r[name]) if r[name] not in ("", "nan") else math.nan for r in rs])
    return v[~np.isnan(v)]


def boot(stat, *samples):
    """Point estimate and 95% interval of stat(*means), resampling each sample independently."""
    samples = [np.asarray(s, dtype=float) for s in samples]
    if any(len(s) == 0 for s in samples):
        return None
    rng = np.random.default_rng(0)
    point = stat(*[s.mean() for s in samples])
    means = [rng.choice(s, size=(N_BOOT, len(s)), replace=True).mean(axis=1) for s in samples]
    b = stat(*means)
    return float(point), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def verdict_positive(t):
    if t is None:
        return "not available"
    return "supported" if t[1] > 0 else "contradicted" if t[2] < 0 else "inconclusive"


def verdict_negative(t):
    if t is None:
        return "not available"
    return "supported" if t[2] < 0 else "contradicted" if t[1] > 0 else "inconclusive"


def verdict_includes_zero(t):
    if t is None:
        return "not available"
    return "supported" if t[1] <= 0 <= t[2] else "contradicted"


def fmt(t):
    if t is None:
        return "n/a"
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--metric", default="student_restraint_apple",
                    help="student_restraint_apple (primary) or student_restraint (per encounter)")
    args = ap.parse_args()
    folder = Path(args.folder)
    g = load(folder)
    m = args.metric
    partner = m.replace("student_", "partner_")

    def runs(cond, lam, rho, phase):
        return col(g.get((cond, lam, rho), []), f"{phase}_{m}")

    def teacher_E(lam, rho):
        rs = [r for (c, l, p), v in g.items() if (l, p) == (lam, rho) for r in v]
        v = col(rs, f"full_{partner}")
        return float(v.mean()) if len(v) else math.nan

    out = []

    def add(pred, cell, quantity, t, verdict):
        out.append({"prediction": pred, "cell": cell, "quantity": quantity,
                    "estimate": "" if t is None else t[0], "lo": "" if t is None else t[1],
                    "hi": "" if t is None else t[2], "verdict": verdict})

    gain_minus = lambda vs, a, vs0, a0: (vs - a) - (vs0 - a0)

    # P1: restraint-specific transmission S(rho, lam) = G(rho, lam) - G(0, lam) > 0
    for lam in LAMS:
        for rho in RHOS:
            t = boot(gain_minus, runs("value_shaping", lam, rho, "full"), runs("asocial", lam, rho, "full"),
                     runs("value_shaping", lam, 0.0, "full"), runs("asocial", lam, 0.0, "full"))
            add("P1", f"lam={lam:g} rho={rho:g}", "S = G(rho) - G(0)", t, verdict_positive(t))

    # P2: G(1, lam) > G(0, lam)
    for lam in LAMS:
        t = boot(gain_minus, runs("value_shaping", lam, 1.0, "full"), runs("asocial", lam, 1.0, "full"),
                 runs("value_shaping", lam, 0.0, "full"), runs("asocial", lam, 0.0, "full"))
        add("P2", f"lam={lam:g}", "G(1) - G(0)", t, verdict_positive(t))

    # P3: G(1, lam) - G(0.5, lam) > 0
    for lam in LAMS:
        t = boot(gain_minus, runs("value_shaping", lam, 1.0, "full"), runs("asocial", lam, 1.0, "full"),
                 runs("value_shaping", lam, 0.5, "full"), runs("asocial", lam, 0.5, "full"))
        add("P3", f"lam={lam:g}", "G(1) - G(0.5)", t, verdict_positive(t))

    # P4: CT(rho=1, lam=0) - CT(rho=1, lam=1); predicted interval includes 0
    E0, E1 = teacher_E(0.0, 1.0), teacher_E(1.0, 1.0)
    if E0 > 0 and E1 > 0:
        ct = lambda f, h, a, E: 0.5 * (f - a) / E + 0.5 * (h - a) / E
        t = boot(lambda f0, h0, a0, f1, h1, a1: ct(f0, h0, a0, E0) - ct(f1, h1, a1, E1),
                 runs("value_shaping", 0.0, 1.0, "full"), runs("value_shaping", 0.0, 1.0, "half"),
                 runs("asocial", 0.0, 1.0, "full"),
                 runs("value_shaping", 1.0, 1.0, "full"), runs("value_shaping", 1.0, 1.0, "half"),
                 runs("asocial", 1.0, 1.0, "full"))
    else:
        t = None
    v = verdict_includes_zero(t)
    if t is not None and v == "contradicted":
        v = "deviation: lower CT where costlier (supports H1)" if t[2] < 0 else "contradicted: higher CT where costlier"
    add("P4", "rho=1", "CT(lam=0) - CT(lam=1)", t, v)

    # P5(a): retained after removal R = A_post - A_solo,post > 0
    for lam in LAMS:
        for rho in RHOS:
            t = boot(lambda p, a: p - a, runs("value_shaping", lam, rho, "post"), runs("asocial", lam, rho, "post"))
            add("P5a", f"lam={lam:g} rho={rho:g}", "retained", t, verdict_positive(t))

    # P5(b): change after removal (post - full, paired per run) is lower at lam=0 than at lam=1, rho=1
    def change(lam):
        rs = g.get(("value_shaping", lam, 1.0), [])
        d = [float(r[f"post_{m}"]) - float(r[f"full_{m}"]) for r in rs
             if r[f"post_{m}"] not in ("", "nan") and r[f"full_{m}"] not in ("", "nan")]
        return np.array(d)
    t = boot(lambda d0, d1: d0 - d1, change(0.0), change(1.0))
    add("P5b", "rho=1", "change(lam=0) - change(lam=1)", t, verdict_negative(t))

    # P6: decision biasing CT < 0.10 in every cell; retained at lam in {0, 1/2} includes 0
    for lam in LAMS:
        for rho in RHOS:
            E = teacher_E(lam, rho)
            if E > 0:
                t = boot(lambda f, h, a: 0.5 * (f - a) / E + 0.5 * (h - a) / E,
                         runs("decision_biasing", lam, rho, "full"), runs("decision_biasing", lam, rho, "half"),
                         runs("asocial", lam, rho, "full"))
            else:
                t = None
            v = ("not available" if t is None else "supported" if t[2] < CT_MAX_DB
                 else "contradicted" if t[1] >= CT_MAX_DB else "inconclusive")
            add("P6", f"lam={lam:g} rho={rho:g}", f"CT (decision biasing) < {CT_MAX_DB}", t, v)
    for lam in (0.0, 0.5):
        for rho in RHOS:
            t = boot(lambda p, a: p - a, runs("decision_biasing", lam, rho, "post"), runs("asocial", lam, rho, "post"))
            add("P6", f"lam={lam:g} rho={rho:g}", "retained (decision biasing) includes 0", t, verdict_includes_zero(t))

    # ---- report ----
    print(f"Predictions (PREDICTIONS.md), metric: {m}; 95% bootstrap intervals over runs\n")
    width = max(len(r["quantity"]) for r in out)
    current = None
    for r in out:
        if r["prediction"] != current:
            current = r["prediction"]
            print(f"--- {current}")
        t = None if r["estimate"] == "" else (r["estimate"], r["lo"], r["hi"])
        print(f"   {r['cell']:<16s} {r['quantity']:<{width}s}  {fmt(t):<28s} {r['verdict']}")
    counts = defaultdict(int)
    for r in out:
        counts[r["verdict"].split(":")[0]] += 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    with (folder / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {folder / 'predictions.csv'}")


if __name__ == "__main__":
    main()
