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

## 2026-10-02 — E4: Phase 1 built (learner, teacher training, teacher levels)

- **What:** `agents/` (state representation with wall mask, tabular
  Q-learner with softmax, episode runner with the Section 6.5 evaluation rule,
  `FrozenTeacher` following Eq. 9), `scripts/train_teacher.py` (lone teacher
  at λ = 1; reports whether its restraint per encounter reaches 0.95, a
  PROVISIONAL reliability threshold), `scripts/make_teacher_levels.py`
  (restraint per encounter, Eq. 10, with bootstrap CIs; per step reported as
  secondary). `check_gate.py` now gives no verdict for runs that cover only
  part of the conditions.
- **Checks:** `python -m pytest -q` → 48 passed (18 new).
- **Preliminary measurement of the SCRIPTED levels** (assistant's sandbox;
  seeded, so it should reproduce exactly on the laptop):
  `python scripts/make_teacher_levels.py --teacher scripted` (alone, λ = 1,
  50 episodes)

  | ρ | restraint per encounter [95% CI] | per step | last apples taken/ep | all patches dead |
  |---|---|---|---|---|
  | 0.5 | 0.247 [0.200, 0.290] | 0.475 | 3.60 | 62% |
  | 0.75 | 0.427 [0.372, 0.475] | 0.759 | 3.54 | 58% |
  | 1 | 1.000 [1.000, 1.000] | 1.000 | 0.00 | 0% |

  Adjacent levels are distinct. Realised restraint is far below the nominal ρ
  because Eq. 9 draws at EVERY step a last apple is adjacent, and the teacher
  often waits beside one for several steps (leaving it through a 3-step wait
  at ρ = 0.75 has probability 0.75³ ≈ 0.42). The proposal already uses the
  measured rate as the level. Point for the supervisor: drawing once per
  ENCOUNTER would make the realised rate close to ρ; Eq. 9 as written means
  per step, so that is what is implemented.
- **Next:** train the teacher (seed 0) on the laptop; decide trained vs
  scripted teacher; record the official levels.

## 2026-10-04 — E5: Phase 1 complete — the teacher and its three levels

- **Trained teacher** (student's laptop): `python scripts/train_teacher.py --seed <s>`
  for s = 0, 1, 2 (alone, λ = 1, γ = 0.99, 600 episodes, 50 final evaluation
  episodes, reliability threshold 0.95 restraint per encounter):

  | seed | restraint per encounter | return | last apples taken/ep (late training) | reliable? |
  |---|---|---|---|---|
  | 0 | 0.029 | 22.88 | 3.4–4.0 | no |
  | 1 | 0.000 | 21.32 | 3.8–4.0 | no |
  | 2 | 0.776 | 20.30 | 2.9–3.9 | no |

  All three end near the return of a purely greedy agent (20.18) and about
  half that of the restrained policy (41.04). Learning made the agent greedier:
  at episode 1 (nearly untrained) seed 0 earned 35.95 with restraint 0.739.
- **Seed 2 shows why restraint must be read with outcomes:** a restraint rate
  of 0.776 with about 3 last apples taken per episode and a greedy-level
  return. It often arrives beside a last apple, leaves and returns; each
  arrival is a new encounter, most are declined, but the apple is eventually
  taken and the patch dies.
- **Check that the environment is not at fault** (assistant's sandbox; lone
  agent at λ = 1, 50 episodes, scripted policies):

  | measure | restrained π_R | greedy π_G | difference |
  |---|---|---|---|
  | undiscounted | 41.04 | 20.18 | +20.86 |
  | γ = 0.99, averaged over all steps | 1.836 | 0.250 | +1.59 |
  | γ = 0.99, from the start of the episode | 13.64 | 16.60 | −2.96 |
  | γ = 0.999, from the start of the episode | 27.88 | 19.78 | +8.11 |

  The environment rewards restraint (it doubles the harvest), and the
  learner's own objective, averaged over the states it visits, favours it.
  From the start of an episode, however, greed pays more within a γ = 0.99
  horizon, and the learner's state does not reveal the time in the episode or
  a patch's history. Restraint is therefore profitable but hard to learn alone:
  the immediate +1 for a last apple wins over a cost that arrives hundreds of
  steps later. This is a learning difficulty, not an environment fault.
- **Decision:** the trained teacher is not reliably restrained in any seed, so
  the fallback applies: **the scripted restrained policy π_R is the teacher.**
  (Not tried: a teacher trained with γ = 0.999, for which restraint pays even
  from the start of an episode. Optional; not needed for the study.)
- **Teacher levels, official** (student's laptop):
  `python scripts/make_teacher_levels.py --teacher scripted`
  (alone, λ = 1, 50 evaluation episodes)

  | ρ | restraint per encounter [95% CI] | per step | return | last apples taken/ep | all patches dead |
  |---|---|---|---|---|---|
  | 0.5 | 0.247 [0.200, 0.290] | 0.475 | 26.32 | 3.60 | 62% |
  | 0.75 | 0.427 [0.372, 0.475] | 0.759 | 28.72 | 3.54 | 58% |
  | 1 | 1.000 [1.000, 1.000] | 1.000 | 41.72 | 0.00 | 0% |

  Identical to the preliminary measurement in E4 (reproduced exactly).
  Adjacent levels have non-overlapping 95% CIs. These measured rates are the
  teacher levels for H2.
- **Considered and set aside:** training the teacher among other learners so
  that it learns restraint. Independent learners in commons dilemmas usually
  converge on over-harvesting, the learning difficulty above would remain, and
  a learned teacher would vary in more than restraint, weakening the control
  needed for H2. A teacher trained with a prosocial reward is noted as possible
  future work.
- **Next:** Phase 2, the students.

## 2026-10-04 — E6: review of the gate; three-valued labels; pilot to include λ = ½

- **Review received** (on the 50-seed gate figure). Points and responses:
  1. *All three teacher levels at λ = ½ are relevant (positive cost, dilemma).*
     Agreed. Already quantified with 95% CIs in E2: Δ = 0.295 [0.248, 0.342]
     (ρ = 0.5), 0.290 [0.237, 0.342] (ρ = 0.75), 0.174 [0.120, 0.228] (ρ = 1).
     Gap: the FIGURE showed no uncertainty. Fixed (error bars).
  2. *λ = 1 labelled "no dilemma" although point estimates suggest a fear-driven
     dilemma.* Correct. The code requires each condition's 95% interval to lie
     above 0; at 50 seeds fear was not established, and the label printed
     "no" where it should have said "not established". With 200 seeds (E3)
     fear is established, so λ = 1 is a fear-only dilemma. R_c(0) and R_d(0)
     involve the greedy partner, not the teacher, so this applies at EVERY ρ.
  3. *Dilemma status and teacher-side cost are separate questions.* Agreed;
     they were separate columns in the tables and separate gate conditions
     (A and C), but the figure titles showed only the dilemma label. Fixed.
  4. *The pilot should include λ = ½.* Agreed: λ = ½ is the only setting where
     restraint is both individually costly and collectively valuable.
     **Decision: the pilot uses λ ∈ {0, ½, 1}** (proposed amendment to the
     pilot description: "using value shaping and the asocial control at
     λ ∈ {0, ½, 1} with the fully restrained teacher; λ = ½ is included
     because it is the setting in which restraint is both individually costly
     and collectively valuable").
- **Changes to `scripts/check_gate.py`:**
  - dilemma status is now **yes / no / unclear**: "no" only if a required
    condition is ruled out (interval below 0) or both temptations are ruled
    out; "unclear" when more seeds are needed; the gate lists unclear conditions;
  - Schelling diagrams show 95% bootstrap CIs on every point, and each panel
    title gives the dilemma status AND Δ with its CI;
  - per-pairing means are stored with their CIs in `gate_summary.json`.
  - Test added for the classification (49 passed).
- **Next:** one clean full gate run with 200 seeds, replacing the combination
  of E2 (50 seeds) and E3 (200 seeds, λ = 1 only) as the reported result.

## 2026-10-04 — E7: OFFICIAL gate, one consistent run with 200 seeds

- **Command** (student's laptop): `python scripts/check_gate.py --seeds 200 --out-dir results/gate_200seeds`
- **Result, primary measure (discounted with γ = 0.99, averaged over all steps):**

  | λ | ρ | Rc(1) | Rd(1) | Rc(0) | Rd(0) | Δ [95% CI] | (1) | (2) | fear | greed | dilemma |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 0.5 | 0.682 | 1.648 | 0.144 | 0.948 | 0.966 [0.928, 1.005] | no | yes | yes | yes | no |
  | 0 | 0.75 | 0.701 | 1.652 | 0.144 | 0.948 | 0.951 [0.913, 0.990] | no | yes | yes | yes | no |
  | 0 | 1 | 0.763 | 1.707 | 0.144 | 0.948 | 0.944 [0.902, 0.984] | no | yes | yes | yes | no |
  | ½ | 0.5 | 0.624 | 0.891 | 0.116 | 0.530 | 0.266 [0.241, 0.293] | yes | yes | yes | yes | yes |
  | ½ | 0.75 | 0.652 | 0.896 | 0.116 | 0.530 | 0.244 [0.217, 0.270] | yes | yes | yes | yes | yes |
  | ½ | 1 | 0.768 | 0.943 | 0.116 | 0.530 | 0.175 [0.147, 0.203] | yes | yes | yes | yes | yes |
  | 1 | 0.5 | 0.409 | 0.088 | 0.065 | 0.072 | −0.321 [−0.341, −0.300] | yes | yes | yes | no | yes |
  | 1 | 0.75 | 0.461 | 0.092 | 0.065 | 0.072 | −0.369 [−0.392, −0.346] | yes | yes | yes | no | yes |
  | 1 | 1 | 0.736 | 0.103 | 0.065 | 0.072 | −0.632 [−0.659, −0.604] | yes | yes | yes | no | yes |

  Gate: A yes, B yes, C yes (Δ > 0 also at λ = ½). **PASSES.** No condition
  is unclear on the primary measure. This run replaces E2 + E3 as the
  reported gate result; figure `results/gate_200seeds/schelling_disc_avg.png`.
- **Other measures:** undiscounted, λ = ½ is a dilemma only at ρ = 1 (ρ = 0.5,
  0.75: condition 1 unclear); λ = 1 a dilemma at every ρ. Discounted from the
  start of the episode, no condition is a dilemma.
- **Observations for the analysis:**
  - λ = ½ is a full (fear and greed) dilemma at every teacher level, with
    positive cost Δ = 0.18–0.27: the core condition.
  - λ = 1 is a fear-only dilemma at every teacher level, with Δ < 0.
  - **At λ = ½, Δ falls as the teacher becomes more restrained** (0.27, 0.24,
    0.18 for ρ = 0.5, 0.75, 1): a restrained teacher keeps patches alive, so
    restraint costs the student less. Teacher level (H2) and cost (H1) are
    therefore not independent across conditions; H2 comparisons at fixed λ
    also differ in Δ. To be stated in the analysis; plotting transmission
    against the measured Δ of every condition makes this visible.
- **Settings confirmed:** T = 1500, p̄ = 0.001, γ = 0.99. Phase 1 results are
  unaffected (same environment and settings).

## 2026-10-04 — E8: Phase 2 built (students, batch runs, parameter selection)

- **What:** `agents/social.py` (value shaping: κ added to Q(s_e, a_e) each time
  the teacher is observed acting, with the teacher's state reconstructed from
  the student's own view, unseen cells treated as empty; decision biasing: with
  probability ω, move towards the teacher's last observed position in the
  episode), `scripts/train_student.py` (asocial / value_shaping /
  decision_biasing beside the scripted teacher π_T^ρ; evaluation with the ε rule),
  `scripts/run_batch.py` (parallel, resumable; several κ, ω, λ, ρ and seeds at
  once), `scripts/summarise_students.py` (results across seeds; groups by every
  setting), `scripts/select_social_param.py`, and the viewer can load a trained
  student (`--student-q`).
- **Rule for choosing κ and ω, fixed BEFORE any selection run**
  (`select_social_param.py`): selection at λ = 0 with ρ = 1 on seeds 100–102,
  which are not reused in the main runs. With A the mean restraint per
  encounter, A_solo that of asocial students on the same seeds and E = 1,
  f = (A − A_solo)/(E − A_solo). Partial if 0.25 ≤ f ≤ 0.75; choose the partial
  value with f closest to 0.5; sensitivity value = its scanned neighbour. If no
  value is partial, choose the value closest to the partial range and REPORT
  that transmission was never partial.
- **Scan values:** κ ∈ {0.003, 0.01, 0.03, 0.1} (log-spaced), ω ∈ {0.1, 0.3, 0.5}.
- **Checks:** `python -m pytest -q` → 57 passed (8 new, including the selection
  rule on synthetic data with a known answer). End-to-end smoke run of every
  condition (3 training episodes; numbers not meaningful).
- **Next:** E9 parameter selection; E10 pilot (asocial vs value shaping at
  λ ∈ {0, ½, 1}, ρ = 1, seeds 0–2; λ = ½ added in E6).

## 2026-10-15 — E9: κ and ω chosen (pre-registered rule, seeds 100–102)

- **Command** (student's laptop; interrupted once when the window was
  closed, restarted, finished runs skipped):
  `python scripts/run_batch.py --conditions asocial value_shaping decision_biasing --kappa 0.003 0.01 0.03 0.1 --omega 0.1 0.3 0.5 --lams 0.0 --rho 1.0 --seeds 100 101 102 --out-dir results/selection`
  then `python scripts/select_social_param.py results/selection`
- **Result** (λ = 0, ρ = 1; A_solo = 0.023 from the asocial students; E = 1):

  | mechanism | value | restraint per encounter | f | partial? | last apples taken/ep | return |
  |---|---|---|---|---|---|---|
  | value shaping | κ = 0.003 | 0.422 | 0.408 | yes | 8.48 | 17.95 |
  | value shaping | κ = 0.01 | 0.599 | 0.589 | yes | 3.98 | 13.91 |
  | value shaping | κ = 0.03 | 0.549 | 0.539 | yes | 4.43 | 16.20 |
  | value shaping | κ = 0.1 | 0.977 | 0.976 | no | 0.61 | 14.79 |
  | decision biasing | ω = 0.1 | 0.072 | 0.050 | no | 20.86 | 27.76 |
  | decision biasing | ω = 0.3 | 0.211 | 0.192 | no | 21.67 | 26.09 |
  | decision biasing | ω = 0.5 | 0.369 | 0.354 | yes | 19.75 | 23.09 |

- **Decision (by the rule fixed in E8):** **κ = 0.03** (sensitivity value
  κ = 0.1); **ω = 0.5** (sensitivity value ω = 0.3). Held fixed across every
  condition from now on.
- **Observations:**
  - Value shaping transmits restraint readily: even κ = 0.003 closes about
    40% of the gap between the asocial student and the teacher; κ = 0.1 is
    near-complete.
  - Decision biasing transmits far less, and its students take about 20 last
    apples per episode (value shaping: about 4): following the teacher leads
    them to last apples, which their unchanged values then take. Harmless at
    λ = 0 (patches regrow), destructive at λ = 1.
  - κ = 0.01 (f = 0.59) scored above κ = 0.03 (f = 0.54) although a larger
    bonus should transmit more: most likely noise with 3 seeds. It does not
    change the choice, but shows how variable 3-seed estimates are.
  - The sensitivity value κ = 0.1 is near-complete transmission, so the
    sensitivity check tests much stronger transmission, not a small step.
- **Next:** E10 pilot: asocial vs value shaping (κ = 0.03) at λ ∈ {0, ½, 1},
  ρ = 1, seeds 0–2.

## 2026-10-15 — E10: pilot, asocial vs value shaping (κ = 0.03), ρ = 1, seeds 0–2

- **Command:**
  `python scripts/run_batch.py --conditions asocial value_shaping --kappa 0.03 --lams 0.0 0.5 1.0 --rho 1.0 --seeds 0 1 2 --out-dir results/pilot`
  then `python scripts/summarise_students.py results/pilot`
- **Run time:** 18 runs, 3 in parallel, 82 min. Per run: asocial 11–12 min at
  every λ; value shaping 11 min (λ = 0), 14 min (λ = ½), 21–22 min (λ = 1).
- **Sanity checks passed:** wall-blocked fraction 0; teacher patch deaths 0 in
  every run; simultaneous patch deaths 0.07–0.29 per episode; asocial students
  learn no restraint (0.018–0.028), giving a clean A_solo.
- **Restraint per encounter** (mean, 95% CI over 3 seeds; f = (A − A_solo)/(1 − A_solo)):

  | λ | asocial | value shaping | f | last apples taken/ep (asocial → VS) |
  |---|---|---|---|---|
  | 0 | 0.024 | 0.788 [0.706, 0.871] | 0.78 | 21.97 → 2.56 |
  | ½ | 0.018 | 0.656 [0.592, 0.696] | 0.65 | 13.61 → 2.11 |
  | 1 | 0.028 | 0.784 [0.778, 0.797] | 0.78 | 2.67 → 1.31 |

- **Change in return (value shaping − asocial):**

  | λ | student | teacher | collective |
  |---|---|---|---|
  | 0 | −14.30 | +8.31 | −5.99 |
  | ½ | −7.51 | +9.69 | +2.18 |
  | 1 | +1.79 | +4.92 | +6.71 |

- **Observations (exploratory, 3 seeds):**
  - Value shaping transmits restraint at every λ.
  - Returns match the gate: at λ = 0 restraint is wasteful (student and
    collective lose); at λ = ½ it is a social dilemma (the student loses, the
    collective gains, collective CIs [35.76, 36.66] vs [38.08, 38.94] do not
    overlap); at λ = 1 it pays individually and collectively.
  - Ordering of transmission is λ = ½ lowest, λ = 0 ≈ λ = 1. This is not the
    order H1 predicts, but 3-seed intervals are only the range of the seeds;
    no conclusion about H1.
  - **Open issue: possible passivity.** Value-shaping students spend about half
    of all evaluation steps adjacent to a protected apple (per-step
    opportunities about 25,000–42,000 vs 126–1,385 for asocial students) and
    their longest stationary spell averages 341–363 steps per episode (asocial:
    63–96). Restraint may partly be "parking" beside last apples. This would
    also explain the large own-return loss at λ = 0 and the longer run times.
    To check whether the teacher (π_R) waits similarly, i.e. whether this is
    faithful transmission or an artefact of the bonus.
- **Next:** inspect a value-shaping student with the viewer; depending on the
  answer, add a stationarity measure for the teacher and student before the
  decision-biasing pilot and the full design.
- **Addendum, viewer check** (seed 0, λ = ½, first 600 steps, greedy student;
  illustration only): from step 16 the only apples left are the 4 protected
  ones (550 of 600 steps); teacher still 92% of steps (longest spell 145),
  student 94% (longest 280); the student is beside a protected apple for only
  42 steps, the teacher for 216. The scripted teacher π_R stays still by
  design whenever no harvestable apple exists, and value shaping rewards the
  student for copying the teacher's actions, including STAY. During
  evaluation the 5% random moves are disabled next to a protected apple, so a
  student whose greedy action there is STAY cannot be nudged away until the
  state changes; this may explain the high per-step counts. To confirm over
  all pilot seeds with scripts/stillness_check.py (teacher and student
  stillness, stillness with food in view, steps with nothing to harvest).

## 2026-10-15 — E11: stillness check (are value-shaping students passive?)

- **Why:** E10 found that value-shaping students spend about half of all
  evaluation steps beside a protected apple and stand still for long spells.
  The question was whether this is passivity caused by the bonus, or copied
  waiting (the scripted teacher π_R stays still by design whenever no
  harvestable apple exists).
- **Tool:** new `scripts/stillness_check.py` replays the frozen evaluation of
  every saved student (same 50 evaluation seeds and settings, learning off) and
  measures teacher and student stillness side by side.
- **Command:** `python scripts/stillness_check.py results/pilot`
  (output: `results/pilot/stillness_check.csv`)
- **Result** (means over seeds 0–2; teacher / student; % of steps):

  | condition | λ | nothing to harvest | still | still with food in view | still beside protected apple | longest still spell beside (steps) |
  |---|---|---|---|---|---|---|
  | asocial | 0 | 91.7 | 96.4 / 22.0 | 4.4 / 0.1 | 17.7 / 0.0 | 160 / 0 |
  | asocial | ½ | 93.4 | 96.9 / 35.7 | 3.3 / 0.2 | 13.4 / 0.0 | 123 / 0 |
  | asocial | 1 | 93.0 | 96.6 / 57.3 | 3.4 / 0.1 | 21.8 / 0.0 | 220 / 0 |
  | value shaping κ = 0.03 | 0 | 88.1 | 93.4 / 80.0 | 3.3 / 0.3 | 28.0 / 49.1 | 196 / 358 |
  | value shaping κ = 0.03 | ½ | 85.7 | 93.4 / 88.1 | 6.5 / 0.6 | 29.8 / 41.9 | 227 / 359 |
  | value shaping κ = 0.03 | 1 | 90.3 | 94.5 / 88.0 | 3.7 / 0.6 | 25.5 / 41.1 | 217 / 335 |

  Re-measured student restraint matches the original evaluations
  (asocial 0.044 / 0.017 / 0.027 vs 0.024 / 0.018 / 0.028; value shaping
  0.834 / 0.671 / 0.758 vs 0.788 / 0.656 / 0.784); small differences because
  the teacher's random stream is not saved.
- **Conclusions:**
  - **Not passive.** Value-shaping students almost never stand still with a
    harvestable apple in view (0.3–0.6% of steps; the teacher 3.3–6.5%).
  - Most stillness is the world, not the student: nothing is harvestable for
    86–93% of steps in every condition, and the teacher is still more often
    (93–97%) than any student.
  - The difference from the teacher is *where* students wait: beside a
    protected apple for 41–49% of steps (teacher 26–30%), with spells of about
    335–360 steps. Two likely reasons: at λ > 0 new apples grow only next to
    surviving apples, so waiting beside the last apple is a sensible foraging
    strategy for a learner that cannot see the whole grid (the privileged
    teacher can); and evaluation disables random moves at restraint decisions
    (correctly, so a random move never counts as taking or leaving an apple),
    which lengthens these waits.
  - Consequences: per-step restraint (≈ 0.997) is inflated by this waiting,
    confirming per-encounter restraint as the primary measure (E8). The
    students' loss of return reflects the last apples left, not missed food.
  - Report wording: transmitted restraint takes the form of waiting beside the
    last apple; students do not forgo visible harvestable apples.
- **Next:** decision-biasing pilot (ω = 0.5) at λ ∈ {0, ½, 1}, ρ = 1, seeds 0–2,
  reusing the E10 asocial students; then Phase 3.

## 2026-10-16 — E12: decision-biasing pilot (ω = 0.5), ρ = 1, seeds 0–2

- **Command** (interrupted once after the λ = 0 runs; restart skipped them):
  `python scripts/run_batch.py --conditions decision_biasing --omega 0.5 --lams 0.0 0.5 1.0 --rho 1.0 --seeds 0 1 2 --out-dir results/pilot`
  then `python scripts/summarise_students.py results/pilot` and
  `python scripts/stillness_check.py results/pilot` (all 27 students).
- **Run time per student:** λ = 0 not recorded (interrupted batch), λ = ½ about
  33 min, λ = 1 about 21 min (3 in parallel). The slowest condition so far.
- **Result** (means over 3 seeds; asocial → decision biasing → value shaping):

  | λ | restraint per encounter | last apples taken/ep | encounters (50 eps) | student return | collective return | all patches lost |
  |---|---|---|---|---|---|---|
  | 0 | 0.024 → 0.362 → 0.788 | 21.97 → 18.93 → 2.56 | 1124 → 1483 → 716 | 29.38 → 22.73 → 15.08 | 45.44 → 44.47 → 39.45 | 0.97 → 0.90 → 0.00 |
  | ½ | 0.018 → 0.358 → 0.656 | 13.61 → 13.55 → 2.11 | 691 → 1056 → 305 | 20.95 → 16.64 → 13.43 | 36.23 → 36.37 → 38.41 | 0.89 → 0.97 → 0.00 |
  | 1 | 0.028 → 0.326 → 0.784 | 2.67 → 3.81 → 1.31 | 137 → 283 → 303 | 10.83 → 7.37 → 12.62 | 28.35 → 23.15 → 35.07 | 0.13 → 0.93 → 0.02 |

  Stillness check, decision biasing (teacher / student, % of steps):

  | λ | nothing to harvest | still | still with food in view | still beside protected apple | beside protected apple |
  |---|---|---|---|---|---|
  | 0 | 90.5 | 94.9 / 43.4 | 3.6 / 1.9 | 18.5 / 0.4 | 18.7 / 2.4 |
  | ½ | 91.3 | 96.1 / 33.6 | 4.3 / 1.5 | 13.1 / 0.2 | 13.3 / 1.7 |
  | 1 | 96.1 | 97.5 / 39.4 | 1.3 / 0.7 | 5.4 / 0.1 | 5.5 / 0.4 |

  Re-measured restraint 0.374 / 0.369 / 0.302 (original 0.362 / 0.358 / 0.326).
- **Observations (exploratory, 3 seeds):**
  - Restraint per encounter rises to about 0.33–0.36, but resource outcomes do
    not improve: last apples taken are unchanged at λ = ½ (13.55 vs 13.61) and
    higher at λ = 1 (3.81 vs 2.67 of at most 4 patches), where all patches are
    lost in 93% of episodes (asocial 13%). Collective return is unchanged at
    λ = 0 and ½ and lower at λ = 1 (23.15 vs 28.35). This confirms the
    "signpost" effect anticipated in E9: following the teacher leads the
    student to last apples, which its own values then take.
  - The rate rises because encounters rise (following the teacher brings the
    student to more last apples), not because fewer apples are taken.
  - The ≈ 0.35 rate is flat across λ and close to the selection value (0.369),
    consistent with a mechanical effect of applying the bias at evaluation
    (ω = 0.5 overrides half of all choices, including restraint decisions)
    rather than learned restraint. The H3 teacher-removal phase tests this.
  - Decision-biasing students do not wait beside last apples (< 2.5% of steps;
    value shaping 41–50%). They stand still with food in view slightly more
    often (0.7–1.9%), probably trying to step into the teacher's cell (agent
    blocked 11–14% of steps), still below the teacher's own 1.3–4.3%.
  - Methodological point: a higher restraint rate without fewer last apples
    taken or higher collective return is not better protection of the
    resource. Restraint must always be reported together with last apples
    taken and collective return (proposal §6.6).
- **For the supervisor:** decision biasing is applied during evaluation as
  pre-registered; its measured restraint is largely mechanical. Proposal: keep
  the design and report it, since the post-removal phase measures what was
  learned.
- **Next:** Phase 3 (CT metric, post-removal phase for H3, partner test).

## 2026-10-16 — E13: Phase 3 built (transmission, persistence, partners)

- **New scripts:** `scripts/phase3_eval.py` (measures every trained student) and
  `scripts/summarise_phase3.py` (CT, H2, H3 and partner tables with bootstrap CIs).
- **Measures per student** (learning off unless stated; evaluation seeds
  3,000,000 + i; 5% random moves away from restraint decisions):
  full (channel on throughout, A_full), half (channel off at step 750, A_half),
  off (channel off throughout, before further learning; identical to the partner
  test with its own teacher), post (post-removal phase for H3: 300 further
  learning episodes with the channel off, teacher present, training seeds
  continued; then evaluated with the channel off), partner tests with a greedy
  agent and a naive learner.
- **Analysis:** CT = ½(A_full − A_solo)/E + ½(A_half − A_solo)/E with A_solo from
  asocial students at the same λ and ρ and E the teacher's measured restraint;
  bootstrap resamples social and asocial runs together. H2: A_full − A_solo.
  H3: A_post − A_solo_post (asocial students go through the same removal phase).
  Restraint is always printed next to last apples taken and collective return.
- **Design decisions (pre-registered, PROVISIONAL until the supervisor agrees):**
  - post-removal phase: 300 episodes (half of training);
  - 50 evaluation episodes per measure;
  - naive partner: a fresh asocial Q-learner with the student's
    hyper-parameters, learning across the 50 partner-test episodes (it also
    receives the 5% evaluation random moves);
  - the same teacher random seed for every measure (common random numbers), so
    measures differ only in the deliberate change; consequence used as a check:
    for asocial students full = half = off exactly.
- **Checks on three 30-episode test students (sandbox, not results):** asocial
  full = half = off exactly; value shaping gives the same restraint with the
  channel on and off (learning off), as the proposal states; decision biasing
  drops to 0 with the channel off; CT arithmetic verified by hand.
- **Next:** run Phase 3 on the 27 pilot students:
  `python scripts/phase3_eval.py results/pilot` then
  `python scripts/summarise_phase3.py results/pilot`.

## 2026-10-16 — E14: Phase 3 on the pilot; encounter inflation found; per-apple restraint added

- **Command:** `python scripts/phase3_eval.py results/pilot` (27 students, 300
  removal episodes, 50 evaluation episodes per measure), then
  `python scripts/summarise_phase3.py results/pilot`.
  Check `asocial full = half = off`: OK.
- **Result, restraint per encounter (Eq. 10)** (means over 3 seeds, 95% bootstrap CI):

  | condition | λ | A_full | A_half | off | post | CT | retained after removal |
  |---|---|---|---|---|---|---|---|
  | value shaping κ = 0.03 | 0 | 0.834 | 0.834 | 0.834 | 0.742 | 0.790 [0.706, 0.875] | 0.705 [0.421, 0.873] |
  | value shaping κ = 0.03 | ½ | 0.671 | 0.671 | 0.671 | 0.553 | 0.653 [0.630, 0.675] | 0.482 [0.349, 0.604] |
  | value shaping κ = 0.03 | 1 | 0.758 | 0.758 | 0.758 | 0.777 | 0.732 [0.645, 0.800] | 0.765 [0.631, 0.881] |
  | decision biasing ω = 0.5 | 0 | 0.374 | 0.270 | 0.171 | 0.080 | 0.279 [0.199, 0.339] | 0.044 [−0.032, 0.106] |
  | decision biasing ω = 0.5 | ½ | 0.369 | 0.261 | 0.003 | 0.016 | 0.298 [0.278, 0.318] | −0.056 [−0.186, 0.020] |
  | decision biasing ω = 0.5 | 1 | 0.302 | 0.274 | 0.000 | 0.947 | 0.261 [0.222, 0.301] | 0.935 [0.878, 0.972] |

  Asocial post-removal restraint 0.037 / 0.072 / 0.012 (continued learning alone
  does not produce restraint).
- **Observations (exploratory, 3 seeds):**
  - Value shaping: CT far from 0 at every λ. After removal, restraint persists;
    it erodes most where it is costly (λ = ½: 0.67 → 0.55, last apples taken
    2.0 → 3.3) and not at all where it pays (λ = 1: 0.76 → 0.78).
  - Decision biasing: restraint depends on the teacher being visible (drops at
    the halfway switch, ≈ 0 with the channel off) and is not retained after
    removal at λ = 0 and ½, as H3 predicts.
  - Partner test: value-shaping students keep restraint with a naive partner
    (0.51–0.88), less with a greedy partner at λ > 0 (0.33, 0.48).
- **Anomaly: decision biasing at λ = 1 after removal (0.947).** Encounter counts
  per episode were 64.5, 121.5 and 17.1 (asocial at λ = 1: 2.7–3.1; value
  shaping: 4–8), with 2.00, 2.00 and 1.88 last apples taken per episode. These
  students move back and forth beside last apples, and each return starts a new
  encounter, so the per-encounter rate is inflated. Part of the effect is real:
  last apples taken fall from 3.4–3.9 (before removal) to about 2 per episode,
  and collective return rises (32.8 vs 29.1 for asocial students after removal).
  Milder inflation also appears for value shaping at λ = 0, seed 1
  (35 encounters per episode vs about 10 for the other seeds).
- **Video check** (decision biasing, λ = 1, seed 0, post-removal Q-table, greedy,
  600 steps; illustration): the student alternates between two vertically
  adjacent cells for long spells ((0,5)↔(1,5) about 120 steps; (0,2)↔(1,2)
  about 300 steps, on the edge of a patch), visits only 17 cells and harvests 1
  apple (return 1; teacher 25). A deterministic 2-step loop caused by state
  aliasing in the 3-feature state; beside a live last apple each return
  starts a new encounter, which explains the inflated per-encounter rate.
  Without the 5% evaluation random moves the student stays trapped in such
  loops; with them it breaks out often enough to reach a return of about 9.
- **Measurement amendment (before the full design):** added restraint PER APPLE:
  each last apple (one patch's spell in the last-apple state, until the apple is
  taken or the patch regrows) counts once per agent, however often the agent
  returns; last apples taken by the other agent are excluded, as pre-empted
  encounters are in Eq. 10. Implemented in `phase3_eval.py` (AppleTracker,
  wrapping env.step; no change to the environment). Verified against the
  environment on scripted policies at every λ: last apples taken match the
  environment's protected-harvest count exactly, and distinct apples met lie
  between apples taken and Eq. 10 encounters. A bug found by this check (a take
  and a regrowth in the same step merged two last apples) was fixed.
  `summarise_phase3.py` now reports CT, H2 and H3 for both measures.
  `--reuse-post` re-measures using the saved post-removal Q-tables.
- **For the supervisor:** propose reporting both measures, with per-apple
  restraint as the primary measure in the full design, since the pilot showed
  that per-encounter restraint is inflated by back-and-forth movement. This is a
  change to the pre-registered primary outcome, made before the full design and
  motivated by a pilot finding, not by the hypotheses.
- **Next:** re-measure the pilot with both measures:
  `python scripts/phase3_eval.py results/pilot --force --reuse-post` then
  `python scripts/summarise_phase3.py results/pilot`.

## 2026-10-16 — E15: pilot re-measured with per-apple restraint

- **Command:** `python scripts/phase3_eval.py results/pilot --force --reuse-post`
  (27 ok, 51 min; the saved post-removal Q-tables were reused), then
  `python scripts/summarise_phase3.py results/pilot`. Check
  `asocial full = half = off`: OK.
- **Result, restraint per apple** (means over 3 seeds, 95% bootstrap CI; per
  encounter in brackets):

  | condition | λ | A_full | post-removal | CT | retained after removal |
  |---|---|---|---|---|---|
  | asocial | 0 / ½ / 1 | 0.002 / 0.001 / 0.022 | 0.003 / 0.004 / 0.012 | | |
  | value shaping | 0 | 0.652 (0.834) | 0.414 (0.742) | 0.650 [0.565, 0.727] | 0.411 [0.243, 0.500] |
  | value shaping | ½ | 0.622 (0.671) | 0.462 (0.553) | 0.621 [0.576, 0.654] | 0.458 [0.439, 0.484] |
  | value shaping | 1 | 0.687 (0.758) | 0.745 (0.777) | 0.665 [0.601, 0.736] | 0.733 [0.614, 0.828] |
  | decision biasing | 0 | 0.053 (0.374) | 0.048 (0.080) | 0.038 [0.033, 0.045] | 0.045 [−0.002, 0.125] |
  | decision biasing | ½ | 0.038 (0.369) | 0.015 (0.016) | 0.033 [0.023, 0.039] | 0.011 [−0.004, 0.023] |
  | decision biasing | 1 | 0.055 (0.302) | 0.564 (0.947) | 0.026 [−0.003, 0.048] | 0.552 [0.401, 0.671] |

  Partner test, value shaping, per apple (λ = 0 / ½ / 1): own teacher
  0.652 / 0.622 / 0.687; greedy agent 0.308 / 0.158 / 0.211; naive learner
  0.558 / 0.417 / 0.359.
- **Conclusions (exploratory, 3 seeds):**
  - **Value shaping:** transmission is robust to the measure (per apple
    0.62–0.69 vs 0.67–0.83 per encounter); CT intervals far from 0 at every λ.
  - **Decision biasing:** per apple, transmission is negligible
    (CT 0.026–0.038; the λ = 1 interval includes 0). Its students leave a last
    apple to follow the teacher and come back to take it later; the
    per-encounter rate (≈ 0.35) counted each departure as restraint. This
    agrees with the outcome measures (last apples taken did not fall).
  - **Persistence ordered by cost:** after removal, value-shaping restraint
    per apple changes by −0.24 at λ = 0 (Δ = +0.94), −0.16 at λ = ½
    (Δ = +0.18) and +0.06 at λ = 1 (Δ = −0.63). The more restraint costs the
    student, the faster its own reward erodes it. A pattern for the full
    design to test, not a finding.
  - **Decision biasing at λ = 1 after removal:** 0.947 per encounter is mostly
    oscillation (E14); 0.564 per apple remains, far above asocial students
    after the same phase (0.012), but produced by looping beside last apples,
    with a lower own return than asocial students (9.04 vs 11.25). Reported as
    an odd case, not as transmission.
  - **Partners:** value-shaping restraint is lower beside a greedy agent or a
    naive learner than beside the teacher; per apple shows this more clearly
    than per encounter.
- **Decision (pending supervisor):** use restraint per apple as the primary
  measure in the full design and report per encounter alongside (amendment
  proposed in E14, motivated by the pilot).
- **README:** pilot results updated with per-apple values and intervals.
- **Next:** supervisor update (findings, the measurement amendment, the three
  provisional Phase 3 settings), then plan the full design (three teacher
  levels, number of runs from the pilot's run times and variability).
