# Scientific Experimental Design

## 11. Scientific Experimental Design

Each central benchmark is considered a complete experiment object.

An experiment defines:

```text
Game Configuration + Bet Abstraction + Bot Version +
Opponent Population + Population Distribution + Population Version +
Random Seed(s) + RNG Configuration +
Sample Size + Power Target, if hypothesis test + Precision Target +
Statistical Method
```

---

### 11.1 Statistical Reporting Hierarchy

Every benchmark distinguishes the following levels:

```text
Hands ↓ Runs / trajectories ↓ Opponent profiles ↓ Population estimator ↓ Agent-level aggregate ↓ Agent family / variant structure
```

No lower-level observation is automatically promoted to an independent observation at a higher level. For adaptive agents, the trajectory/run is the primary independent unit. For H5, independent agent variants are the relevant scientific observations; multiple seeds of the same trained policy are not automatically treated as fully independent agents.

### 11.2 Benchmark Matrix

At a minimum:

| Agent | Nit | TAG | LAG | Calling Station | Maniac | Randomized Population | Stress Population | Holdout |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Hand Strength | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MCTS | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Exploitative | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Adaptive | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

The actual sample sizes are determined according to power and precision targets.

The holdout results are interpreted as final test results only after the agent configuration has been finalised.

---

### 11.3 Ablation Studies

To assess the contribution of individual components, targeted ablations are performed:

1. Without opponent modeling
2. With opponent modeling
3. Without position compensation
4. With position compensation
5. Small bet abstraction
6. Baseline bet abstraction
7. Expanded bet abstraction
8. Different RNG configurations
9. Different populations
10. Different adaptation horizons
11. Different policy‑calibration tolerances
12. Different holdout populations

This prevents an observed improvement from being merely due to an uncontrolled change in experimental conditions.

---
