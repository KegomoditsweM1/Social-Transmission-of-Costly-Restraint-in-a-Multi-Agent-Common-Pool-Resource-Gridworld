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
