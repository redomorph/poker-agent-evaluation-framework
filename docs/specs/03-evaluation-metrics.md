# Evaluation Metrics, Robustness, Exploitability, Power, and Precision

## 3. Evaluation Model

The previous one-dimensional metric "win rate" is discarded.

Instead, three **complementary evaluation dimensions** are used:

| Dimension | Definition | Primary Metric |
| :--- | :--- | :--- |
| **Population Performance** | Expected value against a defined opponent population | empirical EV estimator / 100 in BB |
| **Robustness** | Stability of EV across a distribution of plausible opponents | Mean, Median, 10%-Quantile |
| **Game-Theoretic Quality** | Resistance against Best Response in an explicitly defined game environment | Exploitability / Distance to game value |

The dimensions are deliberately **not called mathematically orthogonal**. Correlations may exist between them. Precisely these interactions are the subject of investigation.

---

### 3.1 Population Performance

For an agent $\pi$ and an opponent parameter vector $\theta$, let $EV(\pi,\theta)$ be the expected net chip win of the agent per hand.

The performance over an opponent population $D$ is defined as:

$$
P(\pi\mid D)=\mathbb{E}_{\theta\sim D}[\mu(\pi,\theta)]
$$

For simulated data, a distinction is made between the theoretical expectation and its empirical estimator.

The empirical EV estimator is:

$$
\widehat{EV}_{100} = 100 \cdot \frac{1}{N} \sum_{i=1}^{N} Profit_i
$$

where $Profit_i$ is the change in the agent's net chip stack in hand $i$ in big blinds.

The symbol $\widehat{EV}_{100}$ therefore denotes an **empirical estimator of the expected net chip win**, not an immediately computed game-theoretic decision EV.

All performance figures are reported in BB/100 unless another unit is explicitly stated. The realised profit of player A in hand $i$ is $\mathrm{Profit}_i=S_{i,after}-S_{i,before},$ where $S_{i,before}$ is the stack immediately **before posting blinds** and $S_{i,after}$ is the stack after the hand. The empirical mean-payoff estimator is

$$
\widehat{\mu}_{100}=100\cdot\frac{1}{N}\sum_{i=1}^{N}\mathrm{Profit}_i.
$$

This empirical mean payoff may be reported as empirical EV/100 for readability, but it is not automatically a decision EV or game-theoretic EV.

---

### 3.2 Robustness

Robustness should not be confused with a theoretical "worst case".

The true worst case would be:

$$
R_{min}(\pi) = \min_{\theta \in \Theta} EV(\pi,\theta)
$$

For a sample from an opponent population, the 10%-quantile is used instead:

$$
R_{10}(\pi\mid D)=Q_{0.10}\left(\widehat{\mu}_{100}(\pi,\theta):\theta\sim D\right)
$$

The 10%-quantile is computed over the opponent parameters ( $\theta \sim D$ ), not over individual hands. For each $\theta$, a stable EV estimator is computed, and these EV values are quantiled over the $\theta$ -distribution.

Additionally, the following values are reported:

- Mean EV
- Median EV
- 10%-quantile
- 25%-quantile
- Standard deviation
- Empirical minimum value (always reported descriptively; it is not interpreted as a guaranteed worst case and is not conditionally suppressed as an outlier)

The 10%-quantile is explicitly referred to as a **robustness metric** and not as a mathematical worst case. A confidence interval for $R_{10}$ is reported using a profile-level or nested bootstrap that respects the hierarchy of opponent profiles and simulation runs. For important results, the preferred procedure is: (1) resample opponent profiles, (2) if repeated runs exist per profile, resample runs within profiles, (3) recompute the profile-level EV, and (4) recompute the 10%-quantile. Small profile counts are flagged as insufficient for precise tail estimation. The empirical minimum is always reported descriptively and is not conditionally suppressed as an outlier.

---

### 3.3 Exploitability

Exploitability is only referred to as such where a defined best-response computation and a reference value of the game are available.

For a game with game value $V^*$, we define: $X(\pi) = BR(\pi) - V^*$ where $BR(\pi)$ denotes the maximum EV achievable by an optimal best-response opponent against $\pi$.

**For Leduc Hold'em:**
- $V^*_{Leduc} = 0$ is used for the defined symmetric zero-sum benchmark environment.
- $X_{Leduc}(\pi)$ is reported in millibigblinds per hand (mBB/h).
- $BR(\pi)$ is computed via the defined OpenSpiel reference environment.
- The exact computation and configuration of the best response is versioned and stored.
- Native OpenSpiel utility units and any conversion to mBB/hand are documented separately. mBB/hand is used only when an explicit Leduc blind/stack convention and conversion factor exist; otherwise native utility units are reported.

**For NLHE:**
- $BR(\pi)$ is not computed in the baseline experiment.
- Exploitability in NLHE is therefore not referred to as such.

---

### 3.4 H5: Statistical Analysis of the Leduc–NLHE Relationship

H5 is formulated as a **predictive/associational analysis**, not as a causal claim. A correlation between Leduc exploitability and NLHE performance does not establish that Leduc exploitability causes NLHE performance or that game-theoretic properties transfer causally between the environments.

The original idea of correlating each individual $EV(\pi,\theta)$ measurement with the same $X_{Leduc}(\pi)$ would not correctly account for the nested data structure.

Therefore, H5 is primarily evaluated at the **agent level**.

For each agent, we first aggregate over the defined test population:

$$
\overline{EV}_{NLHE}(\pi) = \mathbb{E}_{\theta\sim D_{test}}[ EV(\pi,\theta) ]
$$

and: $R_{10}(\pi) = Q_{0.10}[ EV(\pi,\theta) ]$ Then the following relationships are examined:

$$
corr\left( X_{Leduc}(\pi), \overline{EV}_{NLHE}(\pi) \right)
$$

and separately:

$$
corr\left( X_{Leduc}(\pi), R_{10}(\pi) \right)
$$

Thus, **one observation corresponds to one agent**, and no artificial duplication of the same Leduc exploitability over many opponent samples occurs.

H5 requires at least $N_{min}=12$ independent agent variants for a meaningful rank-correlation analysis. Below this threshold, H5 is reported as exploratory with an explicit sample-size limitation. This is a pragmatic project criterion, not a universal statistical requirement.

The association is reported for each previously defined population separately and over the entire set of independent agent variants. Agent family and variant identifiers are retained so that multiple seeds or variants of the same trained policy are not silently treated as independent scientific observations.

If a later extension models individual $\theta$ -samples directly, a hierarchical or mixed-effects model that explicitly accounts for the nesting "opponent sample within agent" will be used.

For aggregation, at least 10 independent opponent samples ( $\theta \sim D_{test}$ ) are used per agent. Each sample itself comprises a sufficient number of hands to obtain a stable $\widehat{EV}_{100}$ estimate (see power analysis, 3.5). The number of samples is documented and varied in sensitivity analyses.

---

### 3.5 Statistical Power Analysis

Before any inferential statistical comparison, a power analysis is performed, provided that a concrete hypothesis test is planned.

**Parameters:**
- $\alpha = 0.05$ (significance level)
- $\beta = 0.20$ (power = 80 %)
- expected minimum relevant EV difference $\delta$
- estimated standard deviation of the paired differences $\sigma_d$

For a paired comparison, approximately:

$$
n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \, \sigma_d^2}{\delta^2}
$$

where $n$ denotes the required number of independent pairs.

The computation is thus directly based on the expected difference and variance, not solely on a generic Cohen's $d$ value.

**Pilot study:** Before a major experiment, a pilot study is conducted to estimate the variance of the paired differences.

The pilot size of 1,000–5,000 hands serves solely as an initial estimate and is not a universal statistical standard.

The estimated variance is used to determine the final sample size. If no pilot data are available for a first cold-start power analysis, a conservative planning value of $\sigma_d \approx 100$ BB/100 is used provisionally; the assumption is replaced by pilot-based variance estimates before the final confirmatory experiment.

---

### 3.6 Precision Target and Sample Size

Power analysis and precision target are treated separately.

**Power analysis** answers:

> How many observations are needed to detect a pre‑defined minimum relevant effect with a certain probability?

**Precision analysis** answers:

> How many observations are needed so that the estimate has a desired maximum uncertainty?

For important benchmark comparisons, an additional estimator-specific precision target is defined, for example:

$$
95\% \text{ CI} \approx \widehat{EV} \pm 1 \,\text{BB/100}
$$

The sample size is determined **before the final experiment** based on a pilot variance or a conservative variance estimator. The precision target explicitly records the estimand, target CI half-width, confidence level, statistical unit, and variance estimator. A mean-EV precision target does not automatically guarantee precision of a population 10%-quantile.

A final experiment is **not** stopped iteratively as soon as an ordinary 95% confidence interval randomly falls below a certain width. This avoids unadjusted sequential stopping procedures altering the nominal error probability.

If adaptive stopping is used in a later study, it must be specified as a separate sequential design and evaluated statistically accordingly.

---
