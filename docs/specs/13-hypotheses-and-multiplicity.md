# Scientific Hypotheses and Multiplicity Control

## 14. Scientific Hypotheses

The framework shall not only produce software, but enable testable hypotheses.

### H1 – Variance Reduction

The paired CRN method produces narrower confidence intervals for the same number of played hands compared to independent random experiments.

Specifically, define

$$
VR = 1 - \frac{Var(\widehat{\Delta}_{paired})}{Var(\widehat{\Delta}_{independent})}.
$$

H1 predicts $VR\ge0.30$. The independent baseline uses the same total number of simulated hands but without paired CRNs: the two agents are evaluated in separate runs with distinct seeds and independent chance realisations. The paired design uses the common deal schedule defined in Section 4.1. The 30% threshold is a project-specific hypothesis, not a universal CRN property.

### H2 – Population Dependence

The ranking of agents by population performance differs systematically across opponent populations. The primary analysis uses a pre-specified rank statistic such as Kendall’s $\tau$ or Spearman’s $\rho$, and/or an agent-by-population interaction model.

### H3 – Robustness Trade‑off

Average population performance and robustness are not assumed to be equivalent. The relationship between $\overline{EV}_{NLHE}$ and $R_{10}$ is tested with a pre-specified correlation or regression model; a trade-off is not assumed a priori.

### H4 – Opponent Modeling

An agent with online opponent modeling achieves higher performance against unknown parametric opponents after sufficient observation time than an otherwise identical non‑adaptive agent.

The **100-hand result is a cumulative adaptation analysis over hands 1–100**, not performance in the 100th hand, not performance beginning after 100 hands, and not a post-100-hand steady-state estimate. It is reported as a learning-curve endpoint and should not by itself be treated as the primary confirmatory evidence for a small EV effect.

For the 100-hand endpoint:

$$
\Delta_{100} = \widehat{EV}_{100}^{adaptive} - \widehat{EV}_{100}^{nonadaptive}.
$$

The cumulative estimator is based on hands 1–100. A longer evaluation horizon is used for the primary confirmatory performance comparison once its sample size is determined by the applicable power/precision analysis. Other horizons (10, 20, 50, 100, 500 hands) remain secondary adaptation analyses.

### H5 – Leduc–NLHE Predictive Association

Lower Leduc best-response exploitability is associated with higher robustness and/or performance in the abstracted NLHE environment. H5 is a **predictive/associational** hypothesis: it asks whether Leduc exploitability explains or predicts variation in NLHE outcomes across independent agent variants. It does not assert causal transfer between games.

The primary analysis is performed at the agent-variant level using Spearman rank correlation:

$$
\rho_S = corr_{Spearman}(X_{Leduc}, \overline{EV}_{NLHE})
$$

and separately: $\rho_S = corr_{Spearman}(X_{Leduc}, R_{10})$

The exact primary statistic is frozen before final analysis.

Correlation of individual opponent samples with repeated $X_{Leduc}(\pi)$ is not used as the primary analysis.

### H6 – Abstraction Sensitivity

The measured performance and possibly the ranking of agents may depend on the chosen bet‑sizing abstraction.

### H7 – Policy Calibration Acceptance Criterion

H7 is treated as a validation/acceptance criterion rather than a conventional inferential hypothesis. A parametric opponent profile is accepted only if its required observed behavioural frequencies are compatible with the predefined target tolerances under the fixed calibration environment. This is a prerequisite for interpreting the synthetic opponent profiles as controlled benchmark opponents.

These hypotheses and validation criteria are not assumed as results, but evaluated according to their pre-specified roles.

---

## 15. Statistical Multiplicity and Confirmatory Status

The framework distinguishes **confirmatory** hypotheses from exploratory analyses. The confirmatory family is declared before the final experiment and includes only hypotheses with a pre-specified primary endpoint and inferential procedure. The default project family is H1–H4 if all four are formally tested confirmatorily; H5–H6 remain secondary/exploratory unless explicitly promoted before final data collection. H7 is a validation acceptance criterion and is not included in the inferential multiplicity family.

For the confirmatory family, the project uses Holm correction unless a different multiplicity procedure is explicitly pre-specified. The report stores:

```text
confirmatory_hypotheses
primary_endpoint_per_hypothesis
alpha
family_of_tests
multiplicity_correction
exploratory_analyses
```

Exploratory ablations, alternative populations, and sensitivity analyses are not converted into confirmatory claims solely because a nominal $p<0.05$ is observed.

Every statistical result reports the effect estimate, confidence interval, sample/cluster size, and practical relevance, not significance alone.

---
