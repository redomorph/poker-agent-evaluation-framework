# Expected Deliverables

## 13. Expected Deliverables

### 13.1 Software

1. **Python CLI tool** (`pokerbench`) with:

```text
run
battle
inspect
export
plot
replay
```

---

### 13.2 Benchmark Data

A reproducible benchmark database with several million up to – depending on the statistically required sample size – over 100 million simulated hands.

The final number is not considered a scientific goal in itself.

Crucial are:

- defined precision
- confidence intervals
- independent experiment pairs
- reproducibility
- documented populations
- traceable seed and RNG configuration

---

### 13.3 Scientific Report

Every scientific benchmark table additionally reports, where applicable:

```text
Agent
Population
N profiles
N runs
Hands/profile
Total hands
Statistical unit
Mean EV/100
95% CI
Median EV/100
R10
R10 CI
Seed count
Configuration hash
Bot version
Opponent version
Failed runs
Failure policy
```

Pairwise comparisons additionally report mean paired difference, 95% CI, number of independent pairs/clusters, effect size, practical threshold, p-value if confirmatory, and multiplicity-adjusted p-value if applicable.

The report examines in particular:

- Population performance
- Robustness
- Leduc exploitability
- Relationship between Leduc exploitability and NLHE performance
- Effects of bet‑sizing abstraction
- Effects of opponent modeling
- Trade‑off between exploitative and robust play
- Pareto front of agents
- Policy calibration
- Generalisation to holdout populations
- Failure modes and limitations of the methodology

---

### 13.4 Replay Viewer

A minimalistic HTML output visualises critical hands.

Display:

- Hole cards
- Board
- Position
- Stacks
- Pot
- Action history
- Available actions
- Chosen action
- Estimated hand strength
- Resulting profit
- Estimated decision EV, if applicable

---
