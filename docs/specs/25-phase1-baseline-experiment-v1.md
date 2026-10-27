# Phase-1 Baseline Experiment v1

## Status

This document is normative for `phase1-baseline-experiment-v1`.

The protocol is frozen **after** completion of `phase1-pilot-v1` and **before** any final Phase-1 benchmark hand is executed.

Machine-readable authority:

```text
configs/experiments/phase1-baseline-experiment-v1.json
```

Protocol version:

```text
phase1-baseline-experiment-protocol-v1
```

Seed schedule version:

```text
phase1-final-seeds-v1
```

Precision-plan version:

```text
population-mean-normal-precision-plan-v1
```

---

## 1. Scientific purpose

The purpose of this experiment is the first **final Phase-1 static-baseline measurement** for the two transparent baseline agents:

```text
random-baseline-v1
hand-strength-baseline-v1
```

against the five already-calibrated static opponent profiles:

```text
nit
tag
lag
calling_station
maniac
```

This experiment produces final Phase-1 data, but it is **not a confirmatory hypothesis test**. No minimum relevant effect, alpha, beta, superiority margin, equivalence margin, or non-inferiority margin is frozen for this experiment.

Therefore the final sample size is determined by a pre-data **precision target**, not by a power calculation for a hypothesis test.

Pilot observations are never included in the final estimates.

Ordinary optional stopping is prohibited.

---

## 2. Binding pilot evidence

The final design is based only on the already-completed frozen pilot:

```text
pilot protocol id:
phase1-pilot-v1

pilot protocol hash:
cc589fc65610bca308d8ed739efab848fff8477e88c20043b045d5ad3ee6c0a9

pilot report hash:
088977f01dff06dc1d64146be35ab2b04f8246e30f7c0130e398c29d6cf24fd1

Gate-6 report hash:
ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a
```

The pilot completed:

```text
1,000 / 1,000 jobs
1,000 valid paired units
0 failed paired units
2,000 / 2,000 recorded engine hands
```

The pilot's only permitted contribution to this final experiment is the variance/runtime planning evidence frozen below. The pilot does not contribute any hand, paired unit, EV estimate, confidence interval, robustness estimate, or inferential result to the final benchmark.

The variance estimator remains:

```text
sample-sd-paired-difference-bb100-ddof1-v1
```

---

## 3. Frozen pilot variance evidence

The pilot standard deviations of the paired agent difference, in BB/100 units, are:

| Hero | Opponent | $\hat\sigma_d$ (BB/100) |
|---|---|---:|
| Random | Nit | 4292.897169873873 |
| Random | TAG | 4859.772553558781 |
| Random | LAG | 5683.443494000354 |
| Random | Calling Station | 6220.160834672358 |
| Random | Maniac | 5925.092545021614 |
| Hand Strength | Nit | 1734.9492210424949 |
| Hand Strength | TAG | 1984.1283422921178 |
| Hand Strength | LAG | 2041.0725091916206 |
| Hand Strength | Calling Station | 179.43060896720894 |
| Hand Strength | Maniac | 2225.7360438672185 |

Every cell contains 100 valid pilot paired units.

These values are fixed planning inputs. They are not re-estimated from final data to change the final sample size.

---

## 4. Primary and secondary populations

### 4.1 Primary population

The primary estimand uses the frozen Baseline Synthetic population:

| Profile | Weight |
|---|---:|
| Nit | 0.10 |
| TAG | 0.35 |
| LAG | 0.30 |
| Calling Station | 0.15 |
| Maniac | 0.10 |

Population identity:

```text
baseline-synthetic-static-profiles-v1
```

### 4.2 Secondary descriptive population

The same final cell evidence is also summarized under the frozen Balanced population:

```text
balanced-static-profiles-v1
```

with 20% weight on each static profile.

No additional simulation is performed merely to reweight the same five profile-level results.

---

## 5. Primary precision estimand

The primary precision estimand for each hero is:

```text
Baseline Synthetic population mean EV (BB/100)
```

The frozen confidence level is:

$$
1-\alpha = 0.95.
$$

The frozen target CI half-width is:

$$
h = 50\ \text{BB/100}.
$$

The normal planning quantile is:

$$
z_{0.975}=1.9599639845400536.
$$

This normal approximation is a **sample-size planning rule**. Final reporting continues to use the already-frozen evaluation/statistical reporting contracts.

---

## 6. Precision-target disclosure

The general project specification gives approximately **±1 BB/100** as an **illustrative example** of an important benchmark precision target. It is not a universal mandatory value.

`phase1-baseline-experiment-v1` deliberately does **not** adopt that illustrative value.

The frozen target is:

```text
95% CI half-width = 50 BB/100
```

This is a materially coarser precision target than the illustrative ±1 BB/100 example.

The decision was made only after the preregistered Phase-1 pilot completed and before any final experiment data were generated. The pilot showed that substantially narrower precision targets would require much larger sample sizes under the frozen Python implementation and the frozen exact/10,000-Monte-Carlo equity mechanism.

This is therefore a documented computational/scientific trade-off for the initial Phase-1 measurement milestone.

The resulting experiment **must not be described as providing ±1 BB/100 precision**, and it must not be used to support scientific claims that require resolving differences substantially smaller than the frozen precision scale.

A future higher-precision study requires a new versioned protocol and a newly frozen sample-size plan.

---

## 7. Conversion from paired difference to hero EV variance

The approved paired statistic remains unchanged:

$$
d=\frac{d_{BTN}+d_{BB}}{2}.
$$

In the frozen zero-rake heads-up zero-sum baseline game, the hero's position-compensated EV is one half of the paired agent difference. Therefore the pilot SD used for hero-EV precision planning is:

$$
\sigma_{hero,j}=\frac{\sigma_{d,j}}{2}.
$$

This conversion is versioned as:

```text
hero-ev-sd-is-paired-difference-sd-divided-by-2-v1
```

It does not change the runner's approved paired-difference definition.

---

## 8. Fixed sample-size calculation

Let $w_j$ be the frozen Baseline Synthetic weight for opponent profile $j$. Within one hero, the same number $n$ of independent paired deals is used for every one of the five profiles.

The pilot-planned variance of the weighted population mean is:

$$
\operatorname{Var}(\widehat P)
\approx
\frac{1}{n}
\sum_j
w_j^2
\left(\frac{\hat\sigma_{d,j}}{2}\right)^2.
$$

The raw precision requirement is:

$$
n_{raw}
=
\left(\frac{z_{0.975}}{h}\right)^2
\sum_j
w_j^2
\left(\frac{\hat\sigma_{d,j}}{2}\right)^2.
$$

The fixed sample size is:

$$
n=\lceil n_{raw}\rceil.
$$

### 8.1 Random baseline

For `random-baseline-v1`:

```text
n_raw = 2768.216219839214
n     = 2769 paired deals per profile
```

The planning half-width after rounding is:

```text
49.9929231163604 BB/100
```

Total Random paired units:

```text
5 × 2769 = 13,845
```

### 8.2 Hand-strength baseline

For `hand-strength-baseline-v1`:

```text
n_raw = 360.1581107091833
n     = 361 paired deals per profile
```

The planning half-width after rounding is:

```text
49.94166338592805 BB/100
```

Total Hand Strength paired units:

```text
5 × 361 = 1,805
```

### 8.3 Total final experiment size

```text
15,650 paired_deal independent units
31,300 simulated engine hands
```

These values are fixed before final execution. The final experiment is not extended because an observed ordinary CI happens to be wider than 50 BB/100, and it is not stopped early because an observed ordinary CI happens to become narrower.

---

## 9. Final seed schedule

The final seed schedule is completely disjoint from the pilot schedule.

Pilot seeds were:

```text
6,100,000 ... 6,100,099
```

The final common seed prefix begins at:

```text
6,200,000
```

### 9.1 Random

```text
6,200,000 ... 6,202,768
```

Count:

```text
2,769 seeds
```

### 9.2 Hand Strength

```text
6,200,000 ... 6,200,360
```

Count:

```text
361 seeds
```

The same hero-specific schedule is reused across all five opponent-profile cells. The 361 Hand Strength seeds are a common prefix of the Random schedule, preserving common-random-number comparability across heroes where both have observations.

The deal schedule remains independent of agent RNG. Agent randomness remains event-indexed by:

```text
(bot_id, hand_id, decision_index, purpose)
```

No final seed may overlap the pilot range.

---

## 10. Failure handling

The final experiment retains the established policy:

```text
illegal_action_policy = exclude_run
replacement_seed_policy = forbidden
```

A failed frozen unit remains failed evidence under its original seed and is reported. No replacement seed is generated.

The final execution report must distinguish requested, completed, valid, and failed independent units and both requested and recorded engine-hand counts.

---

## 11. Reporting

### 11.1 Primary

For each hero, report the Baseline Synthetic population mean EV/100 with the frozen statistical reporting methodology.

### 11.2 Secondary descriptive outputs

Using the same final evidence, also report:

```text
Balanced population mean EV/100
per-profile EV/100
population median
R10
Q25
population standard deviation
empirical minimum
```

The 10%-quantile remains a robustness metric, not a mathematical worst case.

---

## 12. Scope limitations and documented incompleteness

This experiment is intentionally narrower than the complete research programme.

It contains only five frozen static opponent profiles. Therefore:

```text
complete D_test population benchmark = false
>=10 independent opponent samples guidance satisfied = false
R10 precise lower-tail benchmark = false
complete Section 11.2 benchmark matrix = false
```

The five-profile R10 must carry the existing tail-precision warning.

Randomized, Stress, and Holdout populations are not executed in this protocol because complete executable distributions/instances for those populations have not yet been frozen. They must not be invented ad hoc during final execution.

Accordingly, `phase1-baseline-experiment-v1` is described as a **scoped static-profile Phase-1 baseline measurement**, not as completion of the entire population-performance programme or the complete Section 11.2 benchmark matrix.

This limitation is separate from the ±50 BB/100 precision disclosure. Both must be retained in downstream reports.

---

## 13. Immutability rule

Once any final experiment hand is executed, none of the following may change under this protocol version:

```text
precision target
pilot report binding
pilot variance inputs
sample-size formula
per-hero sample sizes
population weights
opponent profiles
agent versions
Gate-6 report hash
seed schedule
failure policy
statistical unit
scientific-use classification
scope disclosures
```

Any scientific change requires a new protocol version and new canonical protocol hash.

---

## 14. Next implementation boundary

This specification freezes the final experiment design only.

The next implementation step is a separate final execution/binding layer that:

1. loads and validates this exact canonical protocol;
2. verifies the exact bound pilot report before the first final hand;
3. verifies the bound Gate-6 PASS evidence;
4. binds Git/Poetry/hardware provenance;
5. executes/resumes only the frozen final jobs;
6. persists every match;
7. never replaces failed seeds;
8. produces final statistical and population reports without incorporating pilot observations.
