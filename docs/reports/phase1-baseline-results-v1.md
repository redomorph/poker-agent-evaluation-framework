# Phase-1 Baseline Results v1

## Status

**Phase-1 static-baseline measurement: COMPLETE.**

This document summarizes the frozen `phase1-baseline-experiment-v1` execution. The canonical source evidence remains `data/phase1/final/phase1-baseline-report.json`; this document and the derived JSON/SVG artifacts do not replace that execution report.

## Evidence identity

| Item | Value |
|---|---|
| Final report hash | `ebc5732733e7333262ac4cc96114478cf9ba25055b5d69eea5d04dbec16483ca` |
| Final protocol | `phase1-baseline-experiment-v1` |
| Final protocol hash | `82cfd5d0da934685af8be50a4f2ad13219c4c276534d206260e8479d8e242674` |
| Pilot report hash | `088977f01dff06dc1d64146be35ab2b04f8246e30f7c0130e398c29d6cf24fd1` |
| Gate-6 report hash | `ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a` |
| Final execution binding hash | `ed615714960a60f60f98e15371fbce64ce596087ee4a03aeb5f92a599e985295` |
| Derived results hash | `23fe13fd4f8cc1ef026de49cb58d665a5c98b03cae161eef81aaf5bd102aa75b` |

## Execution completeness

The frozen final workload completed without scientific failures:

- 15,650 / 15,650 jobs completed.
- 15,650 valid paired statistical units.
- 0 failed paired units.
- 31,300 / 31,300 engine hands recorded.
- Pilot observations were **not** included in final estimates.
- Confirmatory inference was **not** performed.

## Precision disclosure

The general methodology uses approximately **±1 BB/100** as an illustrative precision example. This experiment deliberately froze a coarser **95% confidence-interval half-width target of ±50 BB/100** after the preregistered pilot showed that substantially smaller targets were computationally impractical for the frozen Python/equity implementation.

This experiment must therefore **not** be described as providing ±1 BB/100 precision. The ±50 target is a Phase-1 measurement compromise chosen before final data collection.

## Primary Baseline Synthetic results

| Hero | Mean EV (BB/100) | Fixed-profile 95% CI | Half-width | Target | Nested-bootstrap 95% mean CI | R10 (BB/100) |
|---|---:|---:|---:|---|---:|---:|
| Random | -558.717 | [-604.191, -513.243] | 45.474 | **MET** | [-2940.000, 1768.133] | -803.809 |
| Hand Strength | -283.150 | [-336.008, -230.291] | 52.858 | **NOT MET** | [-1369.683, 348.160] | -419.910 |

The fixed-profile interval evaluates the frozen run-level precision target. The nested bootstrap additionally resamples the opponent-profile level and is therefore much wider with only five static profiles. These intervals answer different uncertainty questions and must not be substituted for one another.

## Descriptive comparison

Hand Strength's Baseline Synthetic mean is **275.567 BB/100 higher** than Random's:

$$
-283.149931 - (-558.717100) = 275.567169\;	ext{BB/100}.
$$

Its descriptive R10 is **383.899 BB/100 higher** than Random's. These are descriptive differences only. No confirmatory superiority hypothesis was preregistered, so no formal superiority claim is made.

## Precision outcome

Random met the frozen ±50 BB/100 target with a realized half-width of **45.474 BB/100**.

Hand Strength narrowly missed it with a realized half-width of **52.858 BB/100**, exceeding the target by approximately **2.858 BB/100**. The frozen sample size is not extended after observing this result; the miss is reported as observed.

## Population and robustness limitation

Only five frozen static opponent profiles are represented. Consequently:

- the experiment is a reproducible **static-profile baseline measurement**, not the complete broader $D_{test}$ population benchmark;
- the nested profile-level bootstrap intervals are very wide;
- R10 is descriptive and carries the mandatory five-profile tail-precision warning;
- additional hands against the same five profiles cannot solve the limited opponent-population breadth;
- randomized, stress, validation, and holdout population work requires separately frozen population definitions and later experiments.

## Scientific interpretation

Within the five frozen static profiles, Hand Strength has the better descriptive Baseline Synthetic mean and R10. Both primary mean estimates are negative. The fixed-profile estimates are materially more precise than the five-profile nested population intervals, which demonstrates that **run-level sampling precision and opponent-population uncertainty are distinct problems**.

Phase 1 therefore establishes a reproducible static-baseline measurement pipeline and final baseline observations, but it does not establish broad-population superiority, narrow-equivalence claims, or precise lower-tail robustness.

## Reproducibility artifacts

- Canonical execution evidence: `data/phase1/final/phase1-baseline-report.json`
- Derived machine-readable summary: `data/phase1/final/phase1-baseline-results-v1.json`
- Derived figure: `data/phase1/final/phase1-baseline-results-v1.svg`

The derived files are intentionally small Git-trackable artifacts. The large SQLite database is generated experimental storage and may be kept outside ordinary Git history; if archived externally, its SHA-256 should be recorded alongside the final execution report.

## Phase boundary

**Phase-1 static-baseline measurement is complete.** Further work should begin with a new, explicitly frozen scope (for example, broader opponent-population definitions, holdout/stress designs, or later adaptive/game-theoretic phases) rather than extending this completed final experiment after observing its results.
