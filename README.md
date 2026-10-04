<<<<<<< HEAD
# Social transmission of costly restraint — Phase 0: environment and gate

Code for the research proposal *Social Transmission of Costly Restraint in a
Multi-Agent Common-Pool Resource Gridworld* (v4), Mokwele Kegomoditswe, UCT;
supervisor Asad Jeewa.

The project is built in phases (proposal Figure 1). **This package is Phase 0
only:** the environment, the scripted policies and the validation gate. There
is no learning code yet; the teacher (Phase 1), the students (Phase 2) and the
evaluation (Phase 3) arrive as later packages, each extracted into this folder.

## Contents

```
commons_env/
  env.py          the gridworld (PettingZoo Parallel API)
  scripted.py     pi_R (restrained), pi_G (greedy), TeacherPolicy pi_T^rho (Eq. 9)
  metrics.py      restraint rate (Eq. 10), per encounter and per step
scripts/
  check_gate.py   the validation gate: Schelling diagrams and Delta per lambda x rho
  view.py         graphical viewer (pygame); both agents scripted
tests/
  test_env.py     environment rules (incl. PettingZoo's official API and seed tests)
  test_teacher.py Eq. 9 teacher levels; discounted returns
results/          outputs (keep everything, including failed runs)
EXPERIMENT_LOG.md what was run, why, and what was decided
```

## Install (Windows PowerShell, Python 3.11 or newer)

From inside this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

Expected: `30 passed`. Then:

```powershell
git init
git add .
git commit -m "Phase 0: environment and gate; 30 tests passing"
```

## The environment and where it comes from in the proposal

| What | Proposal | Code |
|---|---|---|
| 10 x 10 grid, four 5-cell apple patches | §6.2 | `env.py`, `DEFAULT_LAYOUT` |
| Move in four directions or stay; +1 for moving onto an apple | §6.2 | `env.py`, `step()` |
| Moves that would leave the grid have no effect | §6.2 | `env.py`, `step()` (blocked) |
| Fixed episode length (truncation, not termination) | §6.2 | `max_steps` |
| Regrowth p(n; λ) = (1 − λ) p̄ + λ p_SJ(n) | §6.2, Eq. 8 | `env.py`, `_regrow()` |
| Apples visible within 4 cells; teacher observed within 2 | §6.2 | `view_radius`, `social_radius` |
| Social channel: teacher's position and last action only | §3 | `env.py`, `_observe()` |
| Restraint opportunity = ENCOUNTER with a last apple | §6.6, Eq. 10 | `env.py`, `step()`; `metrics.restraint_rate` |
| Per-step restraint rate (secondary) | §6.6 | `restraint_rate(..., per="step")` |
| Patch losses by cause (incl. both agents taking the last two) | §6.6 | `episode_stats["patch_deaths_*"]` |
| π_R, π_G | §6.3 | `RestrainedPolicy`, `GreedyPolicy` |
| Teacher levels π_T^ρ, mixing only at restraint opportunities | §6.3, Eq. 9 | `TeacherPolicy` |

## Watch the environment

```powershell
python scripts/view.py                                     # restrained teacher, greedy student, lambda = 1
python scripts/view.py --lam 0.0                           # the no-dilemma control
python scripts/view.py --teacher mixed --rho 0.5           # a teacher at level rho = 0.5
```

SPACE pause, RIGHT one step, UP/DOWN speed, R new episode, S screenshot, ESC quit.

## Run the validation gate (proposal §6.2)

```powershell
python scripts/check_gate.py --seeds 50
```

For every λ and every teacher level ρ, a restrained (π_R) and a greedy (π_G)
scripted student are paired with a greedy partner and with the teacher at level
ρ. Per condition the script reports R_c(1), R_d(1), R_c(0), R_d(0), the greed gap
**Δ = R_d(1) − R_c(1)** with a 95% interval, the dilemma conditions, and the gate
verdict, for three return measures:

- **disc_avg (primary):** discounted with γ = 0.99 from every step, averaged;
- **undiscounted:** total reward;
- **disc_start:** discounted from the start of the episode.

The gate passes if (A) every λ > 0 is a social dilemma with the fully
restrained teacher, (B) λ = 0 is not, and (C) at least one condition has Δ > 0
(also reported for λ > 0 alone). Outputs: `results/gate/`.

If it fails, the episode length and p̄ may be changed (`--max-steps`,
`--p-bar`; the proposal sets both in the pilot); otherwise the fallback design
(§6.8) applies.

## Settings

| parameter | value | status |
|---|---|---|
| T (`max_steps`) | 1500 | candidate; confirmed by the gate |
| p̄ (`p_bar`) | 0.001 | candidate; confirmed by the gate |
| λ | 0, ½, 1 | proposal Table 2 |
| ρ | 0.5, 0.75, 1 | proposal Table 2 |
| γ | 0.99 | the student's discount factor |
| `view_radius` / `social_radius` | 4 / 2 | proposal §6.2 |

## Working rules

- Activate the environment first (`.venv\Scripts\Activate.ps1`); the prompt
  must start with `(.venv)`.
- Use `python -m pip`, never bare `pip`.
- Later phases arrive as zip files: extract them INTO this folder with
  `Expand-Archive <zip> -DestinationPath . -Force`. Never delete or replace the
  folder (that would lose `.venv` and `.git`).
- Run `python -m pytest -q` before every commit and every experiment.

## Next phase

Phase 1 (the teacher) starts once the gate passes with 50 seeds.
=======
# Social-Transmission-of-Costly-Restraint-in-a-Multi-Agent-Common-Pool-Resource-Gridworld
>>>>>>> 254e1e4a62e638346b733b3312a5cafe66b3c6e0
