# Poker Agent Evaluation Framework — Specification Index

This directory contains the **binding project specification** after the monolithic document was split into topical files.

## Authority and precedence

1. The files listed below collectively form the authoritative specification.
2. `00-development-contract.md` defines specification authority, engineering principles, and the rule against inventing unspecified scientific behaviour.
3. A newer explicitly versioned definition supersedes an older definition of the same concept.
4. `static-profiles-v2` supersedes the original incomplete static-profile table.
5. `calibration-opportunities-v2` is the authoritative definition of VPIP, PFR, 3BetFreq, CBetFreq, FoldToCBet, and Aggression opportunities/events for policy calibration.
6. `parametric-policy-v1` in `16-parametric-policy-v1.md` is the authoritative baseline definition of the policy mechanism $\phi$.
7. Files under `archive/` are historical references only and are not authoritative after this split.
8. If two active specification files appear to conflict and no explicit version resolves the conflict, implementation must stop and the ambiguity must be resolved in the specification before scientific benchmarking continues.

## Mathematics notation

Inline mathematical expressions use `$...$` and block expressions use `$$...$$`.

## Active specification files

| File | Primary responsibility | Original source sections |
|---|---|---|
| `00-development-contract.md` | Specification authority, test coverage, type safety, immutable state, performance and traceability | Engineering principles |
| `01-project-scope.md` | Abstract, motivation, research questions | Abstract, 1 |
| `02-game-environment.md` | HU NLHE baseline and bet abstraction | 2, 5 |
| `03-evaluation-metrics.md` | EV, robustness, exploitability, H5, power, precision | 3 |
| `04-rng-and-experimental-design.md` | CRNs, seeds, position pairing, statistical units, bootstrap/effect sizes | 4 |
| `05-opponent-modeling.md` | Parametric policy space, policy mechanism, populations, adaptation | 7 except calibration/static profiles |
| `06-policy-calibration.md` | Opportunity definitions, calibration tolerances, `static-profiles-v2` | 7.2.1, 7.3, 7.5 |
| `07-architecture-and-data.md` | Persistence, action validation, budgets, performance, replay | 8 |
| `08-validation-gates.md` | Test pyramid, engine/policy tests, Phase-0 gate | 6, 9 |
| `09-implementation-and-roadmap.md` | Repository setup, implementation phases, first milestone | 10, 20 |
| `10-scientific-experimental-design.md` | Experiment objects, reporting hierarchy, benchmark matrix, ablations | 11 |
| `11-risks-and-limitations.md` | Risks, failure modes, methodological limitations | 12, 17 |
| `12-deliverables.md` | Software, data, report, replay viewer deliverables | 13 |
| `13-hypotheses-and-multiplicity.md` | H1–H7, confirmatory status, Holm correction | 14, 15 |
| `14-reproducibility.md` | Required experiment metadata, hashes, lockfile state | 16 |
| `15-extensions-and-conclusion.md` | Future extensions and conclusion | 18, 19 |
| `16-parametric-policy-v1.md` | Frozen baseline policy mechanism $\phi$, observation-v2, equity, strength gates, action precedence | New normative amendment |
| `17-public-run-cli-v1.md` | Public `poker run` CLI and Gate-6-bound static paired-run semantics | New normative specification |
| `18-public-battle-cli-v1.md` | Public `poker battle` CLI, random baseline, deterministic multi-seed aggregation | New normative specification |
| `19-public-export-cli-v1.md` | Deterministic SQLite export bundle, normalized tables, manifest and integrity rules | New normative specification |
| `20-public-plot-cli-v1.md` | Read-only plotting of hash-verified battle reports without statistical recomputation | New normative specification |
| `21-hand-strength-baseline-v1.md` | Deterministic static hand-strength measurement baseline | New normative Phase-1 specification |
| `22-population-evaluation-v1.md` | Population EV/robustness evaluator and executable Phase-1 populations | New normative Phase-1 specification |
| `23-phase1-pilot-protocol-v1.md` | Frozen Phase-1 variance/runtime pilot design | New normative Phase-1 specification |
| `24-phase1-pilot-execution-v1.md` | Phase-1 pilot execution, provenance, resume and integrity rules | New normative Phase-1 specification |
| `25-phase1-baseline-experiment-v1.md` | Frozen final Phase-1 static-baseline experiment design | New normative Phase-1 specification |
| `26-phase1-final-execution-v1.md` | Final Phase-1 execution, provenance, resume and integrity rules | New normative Phase-1 specification |
| `27-bayesian-opponent-model-v1.md` | Bayesian posterior mathematics for observable opponent behavior | New normative Phase-2 specification |
| `28-adaptive-learning-protocol-v1.md` | Adaptive evidence ingestion, priors, trajectory resets and statistical unit | New normative Phase-2 specification |
| `29-posterior-to-range-v1.md` | Deterministic posterior-to-preflop-range mapping and strength reference | New normative Phase-2 specification |
| `30-adaptive-hand-strength-policy-v1.md` | Range-conditioned adaptive hand-strength decision policy | New normative Phase-2 specification |
| `31-adaptive-lifecycle-v1.md` | Runner-managed adaptive lifecycle, transition persistence, replay and validation gate | New normative Phase-2 specification |
| `32-phase2-adaptive-pilot-protocol-v1.md` | Frozen first adaptive treatment/control pilot design, learning-curve horizons, matched trajectory contrasts, and H4 planning boundary | New normative Phase-2 specification |
| `33-phase2-adaptive-pilot-execution-v1.md` | Frozen execution authorization, provenance, launch/readiness, persistence, and resume contract for the adaptive pilot | New normative Phase-2 specification |
| `34-phase2-adaptive-pilot-parallel-execution-v1.md` | Frozen four-process parallel execution authority with worker isolation and parent-only persistence for the unchanged adaptive pilot | New normative Phase-2 specification |
| `35-phase2-h4-confirmatory-planning-v1.md` | Frozen H4 confirmatory planning boundary: 500-hand primary endpoint, holdout requirement, practical-effect/precision targets, variance prerequisite, and inferential rule | New normative Phase-2 specification |
| `36-phase2-h4-holdout-population-v1.md` | Frozen H4 holdout/test population: deterministic five-unit parametric sample, equal weights, calibration/acceptance boundary, sealing, and information-boundary rules | New normative Phase-2 specification |
| `37-phase2-h4-holdout-variance-pilot-v1.md` | Frozen replicated 500-hand H4 holdout variance pilot: 10 complete seed-level population clusters, disjoint seeds, variance-only planning, strict replay/failure admission, and deterministic final-H4 sample-size inputs | New normative Phase-2 specification |
| `38-phase2-h4-holdout-variance-execution-v1.md` | Frozen four-process execution authority for the unchanged Spec-37 holdout variance pilot: exact binding, parent-only persistence, crash-safe resume, and non-executing plan/readiness gate | New normative Phase-2 specification |

## Change control

Scientific benchmark rules must not change silently. When a frozen rule changes:

1. update the relevant specification file;
2. increment its version identifier where applicable;
3. add or update tests before implementation;
4. preserve sufficient old metadata for replay and historical interpretation;
5. record the change in Git;
6. do not combine results generated under incompatible versions unless the analysis explicitly models that difference.

See `CHANGELOG.md` for specification-level changes introduced during and after the split.
