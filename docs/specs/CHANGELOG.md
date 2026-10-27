## 2026-08-17 — H4 holdout variance execution authority v1 frozen

`38-phase2-h4-holdout-variance-execution-v1.md` is added as the separate operational execution authority for the unchanged `phase2-h4-holdout-variance-pilot-v1` scientific design from Spec 37. It authorizes a four-worker spawn-compatible process pool only after its own canonical machine-readable execution contract, full validation suite, clean committed scientific tree, and non-executing plan/readiness gate are green.

The execution authority freezes worker isolation, process/RNG invariance, parent-only durable persistence, snapshot-before-SQLite crash recovery, canonical 100-job scheduling, pair and five-unit seed-cluster checkpoints, exact binding/provenance, and `plan` / `status` / `readiness` / `run` / `resume` semantics. Scheduler terminality remains separate from scientific 10/10 cluster admission, and no scheduler state may directly produce `sigma_500` or final H4 sample-size outputs.

Spec 38 changes no holdout unit, seed, arm, horizon, CRN rule, estimator, replay rule, failure rule, or Spec-35 planning formula. No real variance-pilot hand may be generated until the machine-readable Spec-38 contract and readiness gate are green; confirmatory H4 inference, final-H4 data collection, replacement seeds, reduced-sample variance planning, and variance-pilot mean-driven design changes remain prohibited.

## 2026-08-17 — H4 500-hand holdout variance pilot v1 frozen

`37-phase2-h4-holdout-variance-pilot-v1.md` freezes the replicated long-horizon variance study required before the final H4 confirmatory sample size can be calculated. The study uses exactly ten independent master-seed population clusters (`6,400,000` through `6,400,009`), the unchanged five-unit Spec-36 holdout population, both frozen treatment/control arms, two independently reset positional trajectories per arm, and 500 hands per positional trajectory. The resulting fixed matrix contains 100 arm jobs, 50 matched treatment/control unit pairs, and 100,000 engine hands.

Population aggregation is frozen within master seed first: all five equal-weight holdout-unit contrasts form one `D_population,s(500)` value, and only the ten complete seed-level population contrasts enter the ddof=1 `sigma_500` estimator. The campaign is all-or-nothing for scientific planning: any invalid population cluster prevents `sigma_500` and `n_final` from becoming admissible under this version; replacement seeds, reduced complete-case planning, optional extension, and optional stopping are prohibited.

The variance pilot may feed forward only `sigma_500`, runtime/completeness evidence, and the deterministic Spec-35 power/precision sample-size calculation. Confirmatory H4 inference, policy/population tuning, observed-mean design changes, and pooling with the future final H4 dataset remain prohibited. Spec 37 freezes scientific design only; a canonical machine-readable protocol and separate execution authority/readiness gate are still required before any holdout-variance hand is generated.

## 2026-08-17 — H4 holdout population v1 frozen

`36-phase2-h4-holdout-population-v1.md` freezes the concrete holdout/test population required by Spec 35 before any long-horizon H4 variance or confirmatory hand may be generated. The population is a fixed equal-weight set of five neutral parametric opponents produced by a deterministic SHA-256 Latin-hypercube generator over a predeclared six-dimensional support. Its exact five target vectors, 0.2 weights, generation namespace, Gate-6 calibration dependencies, and canonical population hash are frozen before H4 holdout outcomes.

All five holdout vectors must pass the existing independent `policy-calibration-environment-v1` validation tolerances for all six behavioural metrics. Calibration failure invalidates the complete population; replacement units, alternate generation seeds, support narrowing, and acceptance-based redraws are forbidden. The holdout identity and true parameter vector remain controller-side and are prohibited from the adaptive/control hero observation and learning evidence.

The population is shared unchanged by the future replicated 500-hand variance pilot and the final H4 experiment. Population contrasts are aggregated within master seed before across-seed variance or inference. This specification does not authorize H4 variance-pilot or final-H4 execution; the next required protocol is the separately versioned 500-hand holdout variance pilot.

## 2026-08-17 — H4 confirmatory planning v1 frozen

`35-phase2-h4-confirmatory-planning-v1.md` freezes the planning boundary between the completed adaptive pilot and a future final H4 experiment. The primary H4 endpoint is fixed at 500 cumulative hands per positional trajectory, satisfying the active requirement that the confirmatory comparison be longer than the 100-hand learning-curve endpoint. The specification freezes a 50 BB/100 minimum relevant effect, a 50 BB/100 95% precision half-width target, alpha 0.05, beta 0.20, the trajectory-cluster t-test/CI/bootstrap reporting rule, Holm metadata, and strict pilot/final data separation.

The completed adaptive pilot report is bound as planning evidence, but its observed treatment-effect means are prohibited design inputs. Because the pilot has only one 500-hand seed, Spec 35 explicitly rejects it as a long-horizon variance estimate. A separately versioned replicated 500-hand variance pilot is mandatory before the final cluster count can be frozen.

Spec 35 also freezes the final H4 population role as holdout/test. The concrete unseen parametric holdout population is intentionally not invented here; it must be defined in a separate specification before any holdout variance or confirmatory hand is generated. Specs 32–34 remain unchanged and do not authorize final H4 execution.

# Specification Changelog

## 2026-08-14 — Phase-2 adaptive pilot parallel execution v1 frozen

`34-phase2-adaptive-pilot-parallel-execution-v1.md` adds a separate four-worker process-pool execution authority for the unchanged `phase2-adaptive-pilot-v1` design. Spec 33 remains the frozen serial execution authority and is not amended.

Spec 34 changes only operational scheduling: four isolated worker processes may execute independent canonical trajectory jobs while the parent process alone commits immutable arm snapshots, campaign SQLite records, manifest terminal state, and pair checkpoints. Worker/PID/scheduling identity is prohibited from entering seeds, RNG namespaces, deal schedules, match identities, trajectory hashes, agent state, or scientific analysis.

The fixed 50-job / 25-pair / 18,000-hand matrix, all master seeds, treatment/control definitions, opponents, populations, trajectory horizons, CRN claim, failure policy, estimators, and H4 boundary remain unchanged.

## 2026-08-14 — Phase-2 adaptive pilot execution authorization v1 frozen

`33-phase2-adaptive-pilot-execution-v1.md` is added as the separate execution authority for the unchanged `phase2-adaptive-pilot-v1` scientific design. This follows the existing Phase-1 separation between pilot-design and pilot-execution specifications.

Spec 32 remains immutable, including its historical `design_frozen_execution_not_yet_authorized` design-stage marker and payload hash. Spec 33 authorizes execution only after its own machine-readable contract and validation gate are green. It freezes the already-implemented v1 operational boundary: serial execution, exact binding/provenance, plan/status/readiness/run/resume semantics, immutable job configs, atomic checkpoints, fail-closed resume, and exact authorization of the Spec-32 payload hash.

No treatment/control definition, opponent, population, seed, trajectory length, horizon, estimator, failure policy, permitted analysis, or final-H4 field changes in this execution specification.

## 2026-08-14 — `phase2-adaptive-pilot-v1` design frozen

Spec 32 freezes the first non-confirmatory adaptive treatment/control pilot after the Spec-31 lifecycle validation gate. The design compares `adaptive-hand-strength-agent-v1` with its Spec-30 control `hand-strength-baseline-v1` against the five calibrated development profiles while keeping the opponent identity and parameter vector hidden from the agent.

The pilot freezes:

- `trajectory_cluster` as the adaptive statistical unit and symmetric positional reset boundary;
- a fixed two-tier seed schedule using `6,300,000` through `6,300,004`;
- five matched seeds per profile through 100 hands per positional run, with predeclared seed `6,300,000` continuing to 500 hands;
- the 10/20/50/100/500 cumulative horizons and Early/Main/Steady-State segments already required by the active opponent-modeling specifications;
- treatment-minus-control matched trajectory contrasts in BB/100, with variance estimation only at horizons having replicated seeds;
- within-seed population weighting before across-seed variance estimation;
- complete-pair failure handling with no replacement seeds;
- strict separation of pilot evidence from final H4 data and a prohibition on selecting the final primary H4 horizon from the observed treatment-effect direction, magnitude, or p-value.

The design deliberately leaves the final H4 horizon, population, practical-effect threshold, precision target, sample size, alpha/beta, multiplicity binding, and final seed schedule unset. Pilot execution remains prohibited until a canonical machine-readable protocol and its Spec-32 validation gate are implemented tests-first.

## 2026-08-14 — Specification authority index synchronized through `adaptive-lifecycle-v1`

The active-file table in `docs/specs/README.md` is synchronized with the normative specifications already present in the repository through `31-adaptive-lifecycle-v1.md`. This closes an authority/traceability gap under `00-development-contract.md`; it does not alter the scientific contents of those specifications.

The synchronized index now explicitly includes:

- Specs 17–20: public run, battle, export, and plot CLI contracts;
- Specs 21–26: the deterministic hand-strength baseline, population evaluation, Phase-1 pilot, and final Phase-1 experiment/execution contracts;
- Specs 27–31: the Bayesian opponent model, adaptive learning protocol, posterior-to-range mapping, adaptive hand-strength policy, and adaptive lifecycle.

No frozen protocol payload, protocol identifier, protocol version, payload hash, benchmark result, or scientific decision rule changes in this synchronization.

## 2026-08-10 — `parametric-policy-v1` frozen

The previously undefined policy mechanism $\phi$ is now frozen in `16-parametric-policy-v1.md`.

The amendment defines:

- `agent-observation-v2` and `public-action-trace-v1`;
- `uniform-random-equity-v1` and `uniform-random-range-v1`;
- exact/Monte-Carlo equity selection with the 500,000 joint-outcome threshold and 10,000 Monte Carlo samples;
- one action-RNG draw per policy decision and a separate equity RNG namespace;
- `policy-strength-reference-v1` and `policy-strength-gates-v1`;
- strength-ordered high/low quantile gates for all target statistics;
- deterministic preflop/postflop decision precedence;
- deterministic strength-based bet sizing;
- explicit no-4-bet-or-higher baseline limitation;
- `policy-calibration-environment-v1` with separate fit and validation samples;
- mandatory calibration acceptance of all five `static-profiles-v2` profiles before Phase-0 Gate 6 may pass.

The independent validation calibration sample is prohibited from influencing gate fitting.

## 2026-08-10 — Split specification and calibration profile amendment

### Structural change

The original monolithic project specification was split into topical files under `docs/specs/`. The split files collectively become authoritative. The original monolith is retained under `docs/specs/archive/` for traceability only.

### `static-profiles-v2`

The static opponent profiles now define targets for `3BetFreq`, `CBetFreq`, and `FoldToCBet` in addition to VPIP, PFR, and Aggression where applicable.

| Profile | VPIP | PFR | 3BetFreq | CBetFreq | Aggression | FoldToCBet |
|---|---:|---:|---:|---:|---:|---:|
| Nit | 10% | 4% | 2% | 45% | 0.3 | 55% |
| TAG | 22% | 18% | 7% | 65% | 0.8 | 45% |
| LAG | 32% | 26% | 10% | 70% | 0.9 | 38% |
| Calling Station | 50% | 5% | 2% | 35% | don't-care | 15% |
| Maniac | 55% | 45% | 18% | 80% | 1.5 | 25% |

These remain synthetic benchmark targets and are not empirical human-population estimates.

### `calibration-opportunities-v2`

Exact versioned opportunity/event definitions were added for:

- VPIP;
- PFR;
- 3BetFreq;
- CBetFreq;
- FoldToCBet;
- Aggression.

`ALL_IN` is classified by betting effect for aggression measurement: wager-increasing all-ins are aggressive; all-in calls are passive.

### Consistency correction

`CBetFreq_target` was added to the policy-target examples because CBetFreq was already part of the parameter vector and mandatory calibration metrics.

## 2026-08-12 — Public battle CLI and random baseline

### `public-battle-cli-v1`

The roadmap `poker battle --bot-a random --bot-b tag --seeds 10000` deliverable is now frozen as a deterministic synchronous multi-seed batch over the existing paired runner. `--seeds N` denotes the consecutive master-seed schedule beginning at `--seed-start`, with one paired deal per seed by default. Every seed job remains an immutable persisted match and aggregation uses the existing cluster-aware paired statistical evaluator.

### `random-baseline-v1`

The public random measurement baseline is now defined as a static agent that performs exactly one event-indexed `AgentRNG.choice(...)` over the supplied legal-action sequence per decision. It is explicitly distinct from any special low-variance random policy used only by the Phase-0 symmetry gate.

Battle reports use the existing 10,000-resample percentile bootstrap, retain failed units without replacement, distinguish requested from analyzed engine hands, and may be written as canonical hash-verified JSON.

## 2026-08-12 — Public deterministic export bundle

### `public-export-cli-v1`

The required `poker export` deliverable is frozen as a read-only SQLite-to-bundle transformation. A successful export contains canonical `records.jsonl` plus normalized `matches.csv`, `hands.csv`, `decisions.csv`, and `timeouts.csv`, with deterministic ordering, SHA-256 file hashes, and a canonical manifest/bundle hash. Selected persisted matches are integrity-checked before the bundle is atomically committed; existing destinations are never overwritten.

Parquet export is intentionally not defined in v1 because the frozen Poetry environment currently has no Parquet serializer dependency. Adding one changes the lockfile/reproducibility surface and requires a separately versioned serialization contract.
