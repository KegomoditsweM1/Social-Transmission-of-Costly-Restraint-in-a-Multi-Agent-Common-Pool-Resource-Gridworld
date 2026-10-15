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
