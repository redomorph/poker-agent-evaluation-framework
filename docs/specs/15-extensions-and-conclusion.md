# Extensions and Conclusion

## 18. Possible Extensions

After completion of the MVP:

- Extension of bet abstraction
- Sensitivity analysis of different action sets
- Extension to 6‑max
- More realistic stack sizes
- Rake modelling
- Empirically calibrated opponent populations
- Stronger opponent models
- CFR variants
- NFSP and other RL methods
- Cython/Rust port (if the Python engine cannot achieve the required benchmark size, e.g., >100M hands, within acceptable time, e.g., <24 hours; the port would be limited to the core engine – hand ranking, pot calculation – while agent logic remains in Python)
- Acceleration to >1M simulated hands/second
- Comparison with historical poker agents
- Connection to external game environments
- Investigation of non‑stationary opponents
- True agent‑specific adversarial populations
- Hierarchical statistical models for nested opponent/agent data
- Sequential design methods for adaptive experiment runtimes

A connection to real poker rooms, if implemented at all, will be done exclusively in compliance with the respective terms of use and not as part of the scientific baseline benchmark.

---

## 19. Conclusion

The project does not develop a primary winning bot, but a reproducible evaluation framework for poker agents.

The central research question is:

> **What properties make a strategy in an imperfect game not only strong against known opponents, but simultaneously robust, reproducible, and difficult to exploit?**

For this purpose, three dimensions are separated:

1. **Population Performance** – average empirical EV estimator.
2. **Robustness** – lower quantiles over a defined opponent population.
3. **Game‑Theoretic Quality** – exploitability in Leduc.

Particularly relevant is the empirical investigation of whether Leduc exploitability correlates with NLHE performance or robustness.

The correlation is analysed at the agent level to avoid pseudoreplication through repeated opponent samples.

Another central component is the **calibration of the parametric opponents**. VPIP, PFR, 3‑bet, and other parameters are defined as target quantities and subsequently validated against the actually observed frequencies.

This prevents seemingly controlled opponent profiles from actually implementing a different strategy than their parameters suggest.

The first milestone is not a strong agent, but a deterministic, validated, reproducible, and statistically defensible evaluation infrastructure.

The project therefore consistently follows the principle:

> **Methodology before AI. Evaluation Framework first, Bot second.**

---
