# Phase-1 Closure and Phase-2 Readiness Audit

## Audit basis

This audit is performed against the active split specifications after completion of the frozen `phase1-baseline-experiment-v1` static-profile measurement. It does not reopen Phase 0, the public CLI, Gate 6, or the completed Phase-1 final experiment.

## Phase-1 roadmap status

| Requirement | Status | Evidence / interpretation |
|---|---|---|
| Random baseline agent | PASS | `random-baseline-v1` implemented and used in the frozen final Phase-1 experiment. |
| Deterministic hand-strength baseline | PASS | `hand-strength-baseline-v1` implemented and used in the frozen final Phase-1 experiment. |
| Exact equity where feasible | PASS | `uniform-random-equity-v1` exact/Monte-Carlo boundary is frozen and validated. |
| Pot-odds decisions | PASS | Frozen in `hand-strength-baseline-v1`. |
| Versioned equity/range representation | PASS | Frozen by the Phase-1 equity and policy specifications. |
| Full benchmark matrix against the five calibrated static profiles for Phase-1 baseline agents | PASS | Both Phase-1 hero baselines were evaluated against Nit, TAG, LAG, Calling Station, and Maniac with fixed final sample sizes. |
| Measure performance under baseline bet abstraction | PASS | `phase1-baseline-experiment-v1` is explicitly a baseline-abstraction experiment. |
| Action-abstraction sensitivity | DEFERRED BY SPEC | The active specification requires smaller/baseline/expanded action sets but freezes only the baseline multipliers. Exact small/expanded action sets are not defined; executing H6 now would require inventing game semantics. |
| Calibrated parametric opponents | PASS | Gate 6 passed under the frozen calibration environment and is bound into the final experiment. |
| Reporting of hands/runs/profiles/statistical unit/seeds/configuration hashes | PASS | Frozen pilot/final execution reports and Phase-1 result artifacts contain these identities. |

**Conclusion:** the Phase-1 static-baseline measurement is complete. The abstraction-sensitivity branch remains intentionally deferred until small and expanded bet abstractions are separately frozen.

## Broader benchmark matrix status

Section 11.2 contains Randomized, Stress, and Holdout columns and later Phase-2 agents. That complete matrix is **not** a completed Phase-1 deliverable. The active opponent-modeling specification describes the purpose of those populations but does not yet define complete executable Randomized, Stress, or Holdout distributions. Creating them without a new population specification would be an uncontrolled research choice.

The completed five-profile experiment must therefore continue to be described as a scoped static-profile baseline measurement, not as the full `D_test` population study.

## Phase-2 dependency audit

| Phase-2 item | Status before this patch | Dependency judgment |
|---|---|---|
| Bayesian online learner | GENUINE CODE GAP | Observable opportunity semantics already exist; a posterior/update layer can be frozen without selecting an adaptive strategy. |
| VPIP/PFR and further observable-frequency estimation | GENUINE CODE GAP | Same dependency as Bayesian learner. |
| Uncertainty model | GENUINE CODE GAP | Beta posterior distributions are explicitly required conceptually. |
| Imperfect-information MCTS | DEFERRED PENDING SPEC | The roadmap requires an explicitly versioned imperfect-information method, but determinization, information-set tree keys, search RNG domains, and bet/raise rollout response semantics are not yet frozen. |
| Search-depth ablation 1/2/4/8 | BLOCKED | Depends on a frozen MCTS implementation. |
| Adaptive/exploitative action policy | BLOCKED | Depends on Bayesian state plus a separately frozen posterior-to-policy mapping and dynamic range representation. |
| Dynamic preflop ranges | DEFERRED PENDING SPEC | Range-update semantics are not frozen. |
| Dynamic postflop strategy | DEFERRED PENDING SPEC | Response/action mapping is not frozen. |
| OpenSpiel/Leduc integration | NOT IMPLEMENTED | Independent Phase-2 branch; can be specified after the current observable-model foundation. |
| H4 confirmatory analysis | DEFERRED | Requires a frozen adaptive/non-adaptive pair, prior, horizon, endpoint, power/precision plan, and multiplicity declaration. |
| H5 Leduc-NLHE association | DEFERRED | Requires Leduc integration and at least 12 independent agent variants for the planned primary rank analysis. |
| H6 abstraction sensitivity | DEFERRED | Requires exact small/expanded abstraction definitions. |

## Implemented next foundation

This patch freezes and implements `bayesian-opponent-model-v1`.

The learner:

- consumes only `calibration-opportunities-v2` observable evidence counts;
- uses independent Beta-Bernoulli posteriors;
- models aggression through aggressive-action share rather than misclassifying $A/P$ as a Bernoulli probability;
- is immutable and deterministic;
- rejects duplicate/reordered evidence at the update boundary;
- has canonical state persistence and hashing;
- does not define a project-wide prior.

The lack of a default prior is intentional. The prior is an experiment-level scientific choice and must be frozen before the first adaptive-agent benchmark.

## Next dependency-ordered work

After this patch is green, the next specification work should be **Phase-2 adaptive-agent protocol freezing**, not immediate MCTS implementation. At minimum it must freeze:

1. the six prior pairs for the Bayesian learner;
2. how public actions are converted into live per-hand evidence without hidden-state access;
3. which posterior summaries influence decisions;
4. dynamic preflop range representation;
5. posterior-certainty thresholds or continuous adaptation rule;
6. adaptation horizons and reset semantics;
7. paired trajectory statistical unit;
8. H4 primary endpoint and its power/precision plan if H4 is promoted to confirmatory status.

MCTS should be frozen separately because its information-set and rollout semantics are orthogonal to the Bayesian learner and must not be guessed from the high-level roadmap formula alone.
