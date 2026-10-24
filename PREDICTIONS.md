# Full study: predictions and analysis plan

**Written:** 24 October 2026, before any full-study run.
**Status:** fixed once committed. The git commit time is the record that these
predictions came before the results. Any later change is added as a dated
amendment at the end, never by editing the text above it.

This plan is informed by the pilot (E10–E17, seeds 0–2), so the predictions are
not blind to it. The full study tests them on fresh seeds that played no part
in choosing the design, the parameters or the measures.

---

## 1. Design

| Factor | Levels |
|---|---|
| Teacher | ρ ∈ {0.5, 0.75, 1}; measured restraint E = 0.25, 0.43, 1.00 |
| Control teacher | ρ = 0 (moves and waits like π_R, always takes an adjacent last apple; E = 0) |
| Dilemma | λ ∈ {0, ½, 1} |
| Student | asocial, value shaping (κ = 0.03), decision biasing (ω = 0.5); ρ = 0 with asocial and value shaping only |
| Runs | 5 independent seeds per cell: **1000–1004** (never used before: pilot 0–2, parameter selection 100–102, evaluation 3,000,000+) |

That is 135 main runs plus 30 control runs. Every other setting is unchanged
from the pilot: T = 1500, p̄ = 0.001, γ = 0.99, α = 0.1, β = 5, 600 training
episodes, 300 removal episodes, 50 evaluation episodes per measure, ε = 0.05.
κ and ω stay at the values chosen in E9; they are not retuned.

**Order and stopping rule.** Seeds 1000–1002 run first for every cell, then
1003–1004. If the runs are not finished by **29 October 2026**, the analysis uses
the complete seed blocks finished (minimum 3 seeds per cell) and the report says
so. No run is excluded or repeated because of its result; a run that crashes is
repeated with the same seed and the crash is logged.

## 2. Measures

- **Primary measure:** restraint per apple (each last apple counts once per
  agent; apples taken by the other agent excluded). Adopted after the pilot
  showed that restraint per encounter is inflated by looping (E14, E15).
  *Pending supervisor approval.*
- **Reported alongside:** restraint per encounter, last apples taken per
  episode, student and collective return, patch losses by cause.
- **Gain:** G(ρ, λ) = A_full − A_solo, with A_solo from asocial students beside
  the same teacher at the same λ.
- **Restraint-specific transmission:** S(ρ, λ) = G(ρ, λ) − G(0, λ), added after
  the ρ = 0 control (E17) showed a smaller non-specific component.
- **CT** (Eq. 4) for ρ > 0. **Retained after removal:** A_post − A_solo,post.
- All intervals: 95% bootstrap intervals over runs (10,000 resamples), resampling
  student and asocial runs jointly.

## 3. Predictions

A prediction is **supported** if the 95% interval lies entirely on the predicted
side of 0, **contradicted** if it lies entirely on the other side, and
**inconclusive** otherwise.

**P1. Transmission (Q1).** For value shaping, S(ρ, λ) > 0 in every cell with
ρ > 0 (9 cells).

**P2. Control (Q1).** For value shaping, G(1, λ) > G(0, λ) at every λ.

**P3. Teacher level (Q3, H2).** For value shaping, at each λ, G increases with the
teacher's measured restraint: G(1, λ) − G(0.5, λ) > 0.

**P4. Cost and transmission (Q2, H1).** The pilot found similar CT at every λ, so
the prediction is **no clear effect of cost on CT**: at ρ = 1, the difference in
CT between λ = 0 (highest Δ) and λ = 1 (lowest Δ) has an interval that includes 0.
If it is negative (lower CT where restraint is costlier), that is reported as a
deviation from this prediction and as support for the proposal's H1.

**P5. Persistence (Q4, H3).**
- (a) For value shaping, retained restraint > 0 in every cell with ρ > 0.
- (b) Erosion follows cost: at ρ = 1, the change after removal
  (A_post − A_full) is lower at λ = 0 than at λ = 1.

**P6. Decision biasing.** CT per apple < 0.10 in every cell, and retained
restraint at λ ∈ {0, ½} has an interval that includes 0. Its λ = 1 post-removal
behaviour (looping, E14) is described, not tested.

## 4. Analysis

- Run with the existing scripts:
  `phase3_eval.py results/full`, `summarise_phase3.py results/full`,
  `make_figures.py results/full`.
- Before the results are looked at, `summarise_phase3.py` will be extended to
  report S(ρ, λ) and the differences in P3, P4 and P5(b) with joint bootstrap
  intervals. The extension is tested on the pilot data only.
- **Exploratory, labelled as such:** partner tests, the per-encounter measure,
  learning curves, the asocial λ = 1 anomaly beside the ρ = 0 teacher (E17), and
  any pattern not listed above.
- The predictions involve over 30 interval checks (one per cell for P1, P5(a)
  and P6), so at the 95% level one or two could fail by chance alone. The report
  gives every result, not only the supported ones, and judges each prediction
  by the pattern across its cells.

## 5. Commands

```powershell
# block 1: seeds 1000-1002, every cell
python scripts/run_batch.py --conditions asocial value_shaping decision_biasing --kappa 0.03 --omega 0.5 --lams 0.0 0.5 1.0 --rho 0.5 0.75 1.0 --seeds 1000 1001 1002 --out-dir results/full
python scripts/run_batch.py --conditions asocial value_shaping --kappa 0.03 --lams 0.0 0.5 1.0 --rho 0.0 --seeds 1000 1001 1002 --out-dir results/full

# block 2: seeds 1003-1004
python scripts/run_batch.py --conditions asocial value_shaping decision_biasing --kappa 0.03 --omega 0.5 --lams 0.0 0.5 1.0 --rho 0.5 0.75 1.0 --seeds 1003 1004 --out-dir results/full
python scripts/run_batch.py --conditions asocial value_shaping --kappa 0.03 --lams 0.0 0.5 1.0 --rho 0.0 --seeds 1003 1004 --out-dir results/full
```

---

## Amendments

*(none yet)*
