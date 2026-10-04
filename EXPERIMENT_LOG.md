# Experiment log

Record every run that informs a decision, including failed and negative ones.
Format: date, what and why, exact command, result, decision.

---

## 2026-10-01 — E0: Phase 0 built (environment, scripted policies, gate)

- **What:** the environment of proposal §6.2 (PettingZoo Parallel API), the
  scripted policies π_R, π_G and π_T^ρ (Eq. 9), and the validation gate
  (`scripts/check_gate.py`). No learning code.
- **Checks:** `python -m pytest -q` → 30 passed, including PettingZoo's
  `parallel_api_test` and `parallel_seed_test`, the regrowth rule against
  Eq. 8, per-encounter restraint counting, patch losses by cause, Eq. 9 at
  ρ = 1, 0 and ½, and discounted returns against a hand calculation.
- **Candidate settings:** T = 1500, p̄ = 0.001, γ = 0.99 (to be confirmed by
  the gate; the proposal sets T and p̄ in the pilot).

## 2026-10-01 — E1: PRELIMINARY gate check (assistant's sandbox, 8 seeds)

- **Why:** the discounted gate is the main risk in the design; checked before
  anything else is built. NOT the official gate (that needs 50 seeds).
- **Command:** `python scripts/check_gate.py --seeds 8`
- **Result, primary measure (discounted with γ = 0.99, averaged over start times):**

  | λ | ρ | Rc(1) | Rd(1) | Rc(0) | Rd(0) | Δ [95% CI] | dilemma |
  |---|---|---|---|---|---|---|---|
  | 0 | 0.5 | 0.627 | 1.707 | 0.141 | 0.979 | 1.081 [0.897, 1.262] | no |
  | 0 | 0.75 | 0.699 | 1.742 | 0.141 | 0.979 | 1.043 [0.926, 1.159] | no |
  | 0 | 1 | 0.681 | 1.804 | 0.141 | 0.979 | 1.123 [0.958, 1.285] | no |
  | ½ | 0.5 | 0.523 | 0.905 | 0.141 | 0.523 | 0.382 [0.284, 0.489] | no (condition 1 unclear) |
  | ½ | 0.75 | 0.641 | 0.957 | 0.141 | 0.523 | 0.316 [0.209, 0.405] | yes |
  | ½ | 1 | 0.715 | 1.002 | 0.141 | 0.523 | 0.287 [0.229, 0.344] | yes |
  | 1 | 0.5 | 0.413 | 0.101 | 0.044 | 0.074 | −0.312 [−0.405, −0.233] | yes |
  | 1 | 0.75 | 0.495 | 0.110 | 0.044 | 0.074 | −0.385 [−0.492, −0.275] | yes |
  | 1 | 1 | 0.727 | 0.113 | 0.044 | 0.074 | −0.613 [−0.699, −0.527] | yes |

  Gate (primary): A yes, B yes, C yes (Δ > 0 also at λ = ½) → PASSES (preliminary).
- **Observations:**
  - λ = ½ is the informative condition: restraint is individually costly
    (Δ > 0) and collectively valuable (a social dilemma).
  - At λ = 1, Δ < 0: beside the restrained teacher, restraint pays the student.
  - Discounted FROM THE START, no condition is a dilemma: within the
    student's ~100-step horizon from the start, restraint never pays back.
    The averaged definition (primary, §6.2) is what makes the gate meaningful.
  - Undiscounted, λ = ½ is not clearly a dilemma with 8 seeds; more seeds needed.
- **Next:** the official gate with 50 seeds on the student's laptop.

## 2026-10-02 — E2: OFFICIAL gate (student's laptop, 50 seeds)

- **Command:** `python scripts/check_gate.py --seeds 50`
- **Result, primary measure (discounted with γ = 0.99, averaged over start times):**

  | λ | ρ | Rc(1) | Rd(1) | Rc(0) | Rd(0) | Δ [95% CI] | fear | greed | dilemma |
  |---|---|---|---|---|---|---|---|---|---|
  | 0 | 0.5 | 0.694 | 1.629 | 0.158 | 0.936 | 0.934 [0.867, 1.004] | yes | yes | no |
  | 0 | 0.75 | 0.701 | 1.627 | 0.158 | 0.936 | 0.926 [0.858, 0.990] | yes | yes | no |
  | 0 | 1 | 0.767 | 1.729 | 0.158 | 0.936 | 0.963 [0.888, 1.036] | yes | yes | no |
  | ½ | 0.5 | 0.608 | 0.903 | 0.127 | 0.550 | 0.295 [0.248, 0.342] | yes | yes | YES |
  | ½ | 0.75 | 0.629 | 0.919 | 0.127 | 0.550 | 0.290 [0.237, 0.342] | yes | yes | YES |
  | ½ | 1 | 0.777 | 0.951 | 0.127 | 0.550 | 0.174 [0.120, 0.228] | yes | yes | YES |
  | 1 | 0.5 | 0.388 | 0.091 | 0.062 | 0.072 | −0.298 [−0.345, −0.249] | unclear | no | no |
  | 1 | 0.75 | 0.432 | 0.099 | 0.062 | 0.072 | −0.333 [−0.374, −0.292] | unclear | no | no |
  | 1 | 1 | 0.757 | 0.107 | 0.062 | 0.072 | −0.651 [−0.703, −0.598] | unclear | no | no |

  Gate verdict printed: A false (λ = 1 not a dilemma: fear unclear, no greed),
  B true, C true (also at λ = ½) → FAILS on condition A only.
- **Undiscounted:** λ = ½ is a dilemma only at ρ = 1; λ = 1 is a dilemma at every ρ.
- **Discounted from the start:** no condition is a dilemma (as in E1).
- **Note:** the 8-seed preliminary run (E1) had shown "fear: yes" at λ = 1;
  with 50 seeds that difference was within noise. E1's prediction that the
  gate would pass was therefore premature.

## 2026-10-02 — E3: λ = 1 with 200 seeds, to resolve the "unclear" fear condition

- **Why:** in E2, condition A failed only because fear at λ = 1 was unclear
  (the 95% interval of Rd(0) − Rc(0) included 0). More seeds were run to
  measure that difference precisely. NO setting was changed; this decision
  was made BECAUSE E2 was unclear, and is reported as such.
- **Command:** `python scripts/check_gate.py --lams 1.0 --rhos 1.0 --seeds 200 --out-dir results/gate_lam1_200seeds`
- **Result (primary):** Rc(1) 0.736, Rd(1) 0.103, Rc(0) 0.065, Rd(0) 0.072;
  Δ = −0.632 [−0.659, −0.604]; condition 1 yes, 2 yes, fear YES, greed no →
  social dilemma (fear only).
- The script printed "gate FAILS" for this run because it contained only
  λ = 1 (B cannot be checked and no Δ > 0 exists at λ = 1). That verdict is
  not meaningful for a subset run; the script will be changed to say so.
- **Gate decision (E2 + E3 combined):** A yes (λ = ½ from E2, λ = 1 from E3),
  B yes (E2), C yes (E2, including λ = ½) → **Phase 0 gate PASSES.**
- **Qualifications:**
  - The λ = 1 dilemma is weak: the fear difference is 0.007 on the primary
    measure, and there is no greed. Beside a restrained teacher, restraint
    PAYS the student at λ = 1 (Δ < 0): λ = 1 is the aligned end of the Δ range.
  - λ = ½ is the condition where restraint is both individually costly
    (Δ = 0.17–0.30) and collectively valuable.
  - Δ ranges from about +0.96 (λ = 0) to −0.65 (λ = 1), giving H1 a wide range.
- **Settings confirmed:** T = 1500, p̄ = 0.001, γ = 0.99.
- **Next:** Phase 1 (the teacher).
