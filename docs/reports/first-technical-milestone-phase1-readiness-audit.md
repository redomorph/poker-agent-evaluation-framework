# First Technical Milestone and Phase-1 Readiness Audit

## Audit basis

Authority: active `docs/specs/*.md` only. The archived monolith is not normative.

Baseline evidence supplied by the project owner before this audit:

```text
548 passed in 72.14s
```

This report distinguishes implementation gaps from experiment execution and from roadmap items that are intentionally deferred.

## Classification

```text
PASS                  implemented with existing validation evidence
GAP -> IMPLEMENTED    genuine code gap closed by this change set
EXPERIMENT PENDING    required measurement/run, but not missing framework code
DEFERRED BY SPEC      insufficiently frozen or assigned to a later roadmap phase
OUT OF MILESTONE      valid project requirement, not required for current milestone
```

## First Technical Milestone (`09` §20)

| Requirement | Status | Evidence / rationale |
| --- | --- | --- |
| Repository and Poetry environment | PASS | Existing Poetry test workflow and locked project baseline. |
| Canonical configuration + SHA-256 | PASS | Frozen `ExperimentConfig` canonical serialization/hash. |
| Engine rules and bet/raise semantics | PASS | Versioned baseline game/betting rules and regression coverage. |
| Hand evaluation, pot, stacks | PASS | Engine unit/regression gates. |
| Deal schedule + RNG hierarchy | PASS | Deterministic deal schedule and event-indexed `AgentRNG`. |
| Position compensation | PASS | Frozen paired-difference formula and validation. |
| Random-v-random / symmetry / determinism | PASS | Phase-0 validation evidence. |
| Engine regression tests | PASS | Phase-0 Gate 5 contract plus prior regression suite. |
| Hidden-information safety | PASS | Observation boundary and dedicated validation. |
| Illegal action behavior | PASS | Deterministic rejection/exclude-run behavior. |
| Agent budget/timeout policy | PASS | Versioned budget/timeout handling and replay coverage. |
| Policy calibration | PASS | Gate-6 scientific PASS with persisted snapshots/report. |
| Statistical evaluation | PASS | Cluster-aware paired evaluator, bootstrap CI, effect size. |
| Power/precision planning | PASS | Frozen planning module and configuration persistence. |
| Experiment configuration persistence | PASS | SQLite persistence plus immutable config/hash metadata. |
| Replay any match | PASS | Archive/SQLite replay and public `poker replay`. |

**Conclusion:** the First Technical Milestone remains complete. No milestone code was reopened.

## Public software deliverable (`12` §13.1)

| Command | Status |
| --- | --- |
| `poker run` | PASS |
| `poker battle` | PASS |
| `poker inspect` | PASS |
| `poker export` | PASS |
| `poker plot` | PASS |
| `poker replay` | PASS |

## Phase 1: Baselines & Measurement (`09` Phase 1)

| Requirement | Audit status | Action |
| --- | --- | --- |
| Random baseline | PASS | `random-baseline-v1` already implemented. |
| Deterministic hand-strength baseline | GAP -> IMPLEMENTED | Added `hand-strength-baseline-v1`. |
| Exact equity where feasible | PASS | `uniform-random-equity-v1` exact/MC boundary already frozen. |
| Pot-odds decisions | GAP -> IMPLEMENTED | Hand-strength baseline now uses frozen integer pot odds. |
| Versioned equity/range representation | PASS | `uniform-random-equity-v1` / `uniform-random-range-v1`. |
| Full benchmark matrix vs calibrated static profiles | EXPERIMENT PENDING | Existing `PairedRunner` can execute the matrix once sample plans are frozen; running the matrix is data collection, not missing engine code. |
| Measure baseline bet abstraction | EXPERIMENT PENDING | Requires a pre-declared benchmark experiment/sample plan. |
| Action-abstraction sensitivity checks | DEFERRED BY SPEC | Baseline abstraction is frozen; concrete small/expanded alternatives are not yet numerically frozen for a confirmatory run. |
| Recalibrate static opponents | PASS | Gate-6 PASS; no recalibration should occur unless a scientific dependency changes. |
| Report hands/runs/profiles/unit/seeds/hashes | PARTIAL -> IMPLEMENTED FOR POPULATION METRICS | Pairwise battle reporting already covers run/seed/unit/hash fields; new population evaluator adds profile/run/hand counts and population/evidence/report hashes. Full multi-phase scientific report remains later work. |

## Population and robustness requirements (`03` §3.1–3.2, `05` §7.6–7.8)

| Requirement | Audit status | Action |
| --- | --- | --- |
| Population mean EV | GAP -> IMPLEMENTED | Weighted population mean over profile-level EV. |
| Median / Q25 / R10 / SD / empirical minimum | GAP -> IMPLEMENTED | Added frozen weighted inverse-CDF metrics. |
| R10 confidence interval | GAP -> IMPLEMENTED | Added deterministic nested profile/run bootstrap. |
| Flag small tail sample | GAP -> IMPLEMENTED | `N_profiles < 10` reporting flag. |
| Balanced Population | GAP -> IMPLEMENTED | Exact five-profile 20% weights. |
| Baseline Synthetic Population | GAP -> IMPLEMENTED | Exact 10/35/30/15/10 weights. |
| Stress Population | DEFERRED BY SPEC | Current spec gives examples, not a complete frozen finite population. |
| Randomized Population | DEFERRED BY SPEC | Current spec does not provide a complete joint distribution for every parameter. |
| Development/Validation/Holdout concrete instances | DEFERRED BY SPEC | Usage semantics are defined, but concrete population instances/seeds are experiment-specific and not yet frozen. |

## Scientific hypotheses and later phases

| Requirement | Status | Reason not implemented here |
| --- | --- | --- |
| H1 independent non-CRN experiment | OUT OF MILESTONE | Scientific hypothesis execution; primary design/sample plan must be frozen before confirmatory data collection. |
| H2/H3 population-dependence/trade-off inference | OUT OF MILESTONE | Requires completed population benchmark data. |
| H4 Bayesian opponent modeling/adaptation | DEFERRED BY SPEC | Phase 2. |
| H5 Leduc exploitability association | DEFERRED BY SPEC | Phase 2/OpenSpiel plus >=12 independent agent variants. |
| H6 abstraction sensitivity inference | OUT OF MILESTONE | Requires frozen alternative abstractions and experiment. |
| H7 calibration acceptance | PASS | Gate-6 scientific PASS. |
| MCTS, Bayesian learner, exploitative/adaptive agent | DEFERRED BY SPEC | Phase 2. |
| NFSP/meta-optimisation/Pareto front | DEFERRED BY SPEC | Phase 3. |

## Result of the audit

After this change set, there are **no remaining genuine code gaps that block the First Technical Milestone or Phase-1 baseline/measurement foundation**.

The next work is scientific experiment preparation rather than more framework plumbing: freeze a Phase-1 benchmark experiment (agents, populations, seed schedule, sample-size/precision target, and output paths), then execute the matrix. Stress/randomized/holdout populations must be frozen numerically before they can be run; they must not be guessed from the examples in the current specification.
