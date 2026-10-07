"""
Run a batch of student-training runs in parallel (one process per run).

Every combination of --conditions x --lams x --rho x --seeds (with --kappa for
value_shaping, --omega for decision_biasing) becomes one call to
scripts/train_student.py. --rho accepts several teacher levels. Each run writes
its own log file. Runs whose summary file already exists are SKIPPED, so an
interrupted batch can simply be started again.

Usage (from the project folder, virtual environment active):
    python scripts/run_batch.py --conditions value_shaping --kappa 0.01 --lams 0.0 1.0 --seeds 0 1 2 --out-dir results/kappa001
    python scripts/run_batch.py ... --workers 4
    python scripts/run_batch.py ... --dry-run        # list the runs, start nothing

--workers defaults to (number of CPU cores - 1), so the computer stays usable.
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tag_for(cond, kappa, omega, lam, rho, seed):
    if cond == "value_shaping":
        return f"{cond}_k{kappa}_lam{lam}_rho{rho}_s{seed}"
    if cond == "decision_biasing":
        return f"{cond}_w{omega}_lam{lam}_rho{rho}_s{seed}"
    return f"{cond}_lam{lam}_rho{rho}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["asocial", "value_shaping"],
                    choices=["asocial", "value_shaping", "decision_biasing"])
    ap.add_argument("--kappa", type=float, nargs="+", default=[0.01], help="value-shaping bonus(es)")
    ap.add_argument("--omega", type=float, nargs="+", default=[0.3], help="decision-biasing weight(s)")
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--rho", type=float, nargs="+", default=[1.0], help="teacher level(s)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--out-dir", default="results/batch")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    combos = []
    for cond in args.conditions:  # a mechanism's parameter only varies for that mechanism
        params = args.kappa if cond == "value_shaping" else args.omega if cond == "decision_biasing" else [None]
        combos += [(cond, p) for p in params]
    for (cond, param), lam, rho, seed in itertools.product(combos, args.lams, args.rho, args.seeds):
        kappa = param if cond == "value_shaping" else None
        omega = param if cond == "decision_biasing" else None
        tag = tag_for(cond, kappa, omega, lam, rho, seed)
        if (out / f"{tag}_summary.json").exists():
            print(f"skip (already done): {tag}")
            continue
        cmd = [sys.executable, str(HERE / "train_student.py"),
               "--condition", cond, "--lam", str(lam), "--rho", str(rho),
               "--seed", str(seed), "--episodes", str(args.episodes), "--out-dir", str(out)]
        if cond == "value_shaping":
            cmd += ["--kappa", str(kappa)]
        if cond == "decision_biasing":
            cmd += ["--omega", str(omega)]
        jobs.append((tag, cmd))

    print(f"{len(jobs)} runs to do, {args.workers} at a time, output in {out}")
    if args.dry_run or not jobs:
        for tag, cmd in jobs:
            print("  ", " ".join(cmd))
        return

    def run(tag, cmd):
        t0 = time.time()
        with open(out / f"log_{tag}.txt", "w") as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        return tag, code, time.time() - t0

    t_start, failed = time.time(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, tag, cmd) for tag, cmd in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            tag, code, secs = fut.result()
            status = "ok" if code == 0 else f"FAILED (exit {code}; see log_{tag}.txt)"
            if code != 0:
                failed.append(tag)
            print(f"[{i}/{len(jobs)}] {tag}: {status} in {secs / 60:.1f} min "
                  f"(elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)

    print(f"done: {len(jobs) - len(failed)} ok, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
