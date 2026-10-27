# Challenges, Risk Mitigation, and Methodological Limitations

## 12. Challenges & Risk Mitigation

| Challenge | Solution / Workaround |
| :--- | :--- |
| **Large game tree** | Bet‑sizing and possibly card abstraction |
| **Continuous bet space** | Explicitly defined discrete action abstraction |
| **Ambiguous raise semantics** | Formal definition of all bet/raise rules |
| **High poker variance** | CRNs + position pairing + confidence intervals |
| **Statistical spurious precision** | Separate power/precision planning + bootstrap sensitivity analysis |
| **Overfitting to static profiles** | Randomised populations + stress population + holdout |
| **Arbitrary opponent distributions** | Explicit documentation and ablation of different populations |
| **Weak opponent definition** | Formal policy $\pi(a|s,\theta)$ |
| **Policy calibration error** | Calibration tests between target and observed frequencies |
| **Uncertainty in opponent modeling** | Bayesian posterior distributions |
| **Best‑response runtime** | Restriction to Leduc or other exactly controllable games |
| **Misinterpretation of Leduc results** | Separate metrics for Leduc and NLHE |
| **Engine bugs** | Unit, symmetry, regression, and random‑vs‑random tests |
| **Reproducibility** | Seed hierarchy, version pinning, experiment hashes |
| **Overfitting to one bet abstraction** | Ablation of different action sets |
| **Pseudoreplication in H5** | Aggregation at agent level or hierarchical model |
| **Uncontrolled sequential stopping** | Pre‑determined sample size or explicit sequential design |

---

### 12.1 Failure Mode Analysis

The framework may fail in the following scenarios:

| Failure Mode | Detection | Countermeasure |
| :--- | :--- | :--- |
| **Engine bug** | Symmetry/regression test fails | Unit tests, regression tests |
| **Overfitting to population** | Holdout performance collapses | Holdout evaluation, stress population |
| **Seed dependence** | Results vary strongly across seeds | Multiple seeds, report seed variance |
| **Statistical noise** | CI width > relevant effect size | Increase sample size |
| **Model misspecification** | Leduc‑NLHE correlation absent | Reject H5, revise model assumptions |
| **Unrealistic opponents** | Agent performs poorly against real players | Empirical calibration as extension |
| **Policy calibration error** | Observed frequencies deviate systematically from targets | Re‑calibrate policy mechanism |
| **Pseudoreplication** | Results depend heavily on number of opponent samples per agent | Agent level as primary unit; mixed‑effects model for nested analyses |
| **Population leakage** | Holdout results improve during development | Use new sealed test population |
| **Abstraction sensitivity** | Agent ranking changes strongly with action set | Evaluate multiple abstractions |

---

## 17. Limitations of the Methodology

The framework deliberately does not claim to fully represent the strategic quality of a real‑world poker agent.

In particular, the following limitations apply:

1. The NLHE environment uses a discrete bet‑sizing abstraction.
2. The benchmark initially does not use real poker‑room rules.
3. The baseline benchmark uses no rake.
4. Parametric opponents are models and not a complete representation of real players.
5. Leduc exploitability is not a direct NLHE exploitability.
6. Results against a synthetic population cannot be automatically transferred to human players.
7. A large number of simulated hands does not compensate for systematic modelling errors.
8. The synthetic opponent populations cannot capture the strategic diversity of human players.
9. The results initially apply only to the specific bet‑sizing abstraction and cannot be directly transferred to other abstractions.
10. Leduc exploitability is not a substitute for NLHE exploitability; it serves exclusively as a proxy for empirical correlation analyses.
11. The quality of the population performance depends on the quality and calibration of the parametric opponent policy.
12. The holdout evaluation can measure true generalisation only if the holdout population was not used for optimisation during development.
13. A statistically significant effect is not automatically practically or strategically relevant.
14. The correlation between Leduc and NLHE may be limited by the small number of agents examined.
15. CRNs reduce variance, but neither eliminate card variance nor policy variance completely.
16. Benchmark runtime and throughput depend on the physical reference hardware and are not scientific measures of agent quality.

These limitations are explicitly discussed in the scientific report.

---
