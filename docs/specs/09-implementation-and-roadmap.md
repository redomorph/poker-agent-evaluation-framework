# Technical Setup, Implementation Roadmap, and First Milestone

## 10. Initial Technical Project Setup & Implementation Roadmap

The first coding step is deliberately **technical project preparation**, not AI development. The goal is to create a small, reproducible local software project in which the Phase-0 gates can later be implemented without changing the experimental assumptions ad hoc.

### 10.1 Local Project Foundation

The baseline is developed and benchmarked on a **local reference machine**. Cloud instances are not required for the baseline. The exact machine and software environment are recorded for reproducibility.

Recommended initial repository structure:

```text
poker-agent-evaluation/
├── pyproject.toml
├── poetry.lock
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   └── pokerbench/
│       ├── engine/
│       ├── agents/
│       ├── evaluation/
│       ├── experiments/
│       ├── replay/
│       └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── validation/
├── configs/
│   ├── game/
│   ├── agents/
│   ├── populations/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reports/
└── scripts/
```

Poetry is the dependency-management and lockfile tool. The initial environment is created from `pyproject.toml`, committed with `poetry.lock`, and never changed silently between benchmark runs.

### 10.2 First Implementation Sequence

The first implementation sequence is:

1. initialise the Git repository and Poetry project;
2. create the package and test structure;
3. freeze the game configuration as a versioned data object;
4. implement canonical configuration serialisation and SHA-256 hashing;
5. implement the card/deal schedule independently of agent RNG;
6. implement the immutable game state and legal-action generator;
7. implement and freeze the exact bet/raise semantics defined in Section 5.2, including versioning and unit-test coverage, before Phase 0 can be considered complete;
8. implement hand evaluation, pot accounting, stack reset, and position rotation;
9. implement the `TRUE_GAME_STATE` / `AGENT_OBSERVATION` boundary;
10. implement deterministic replay of a single hand;
11. add unit and regression tests before building any sophisticated agent;
12. implement the paired runner and statistical-unit metadata;
13. implement the symmetry and random-vs-random gates;
14. implement policy calibration in the frozen calibration environment;
15. only then implement baseline agents and larger benchmark runs.

### 10.3 Configuration and Reproducibility Contract

Phase 0 uses a **binding technical specification**: rules that affect legal actions, payoffs, information availability, RNG, or statistical interpretation are frozen and versioned before scientific benchmarking.

Every frozen configuration is serialised canonically and hashed with SHA-256. The hash covers the complete relevant configuration rather than merely the game name or seed.

### 10.4 Hardware and Execution Environment

The primary benchmark environment is a local reference machine. For each benchmark, record:

```text
cpu_model
cpu_core_count
cpu_architecture
ram
python_version
os_identifier = platform.platform()
poetry_lock_hash
git_commit
```

A platform-independent hardware fingerprint may be derived from CPU model, core count, and architecture. Exact hardware serial numbers are not required.

### Implementation Roadmap

The development strictly follows the principle:

> **Evaluation Framework first, Bot second.**

---

### Phase 0: Core Infrastructure & Validation (Weeks 1–2)

- [ ] Implement deterministic NLHE engine
- [ ] Freeze canonical configuration and SHA-256 hash
- [ ] Card deck
- [ ] Board development
- [ ] Pot calculation
- [ ] Stack management
- [ ] Hand ranking
- [ ] Defined bet‑sizing abstraction
- [ ] Deterministic pre-defined deal schedule
- [ ] Isolated/event-indexed agent RNG
- [ ] Hidden-information safety boundary
- [ ] `PairedRunner`
- [ ] Statistical evaluation
- [ ] Bootstrap CI
- [ ] Parametric bot factory
- [ ] Explicit separation of policy parameters and observable frequencies
- [ ] Policy calibration test
- [ ] SQLite/Parquet data model
- [ ] Replay data structure
- [ ] Unit tests
- [ ] Symmetry test
- [ ] Random‑vs‑random test
- [ ] Determinism test
- [ ] Regression tests
- [ ] Illegal-action validation
- [ ] Agent budget/timeout policy
- [ ] Calibration environment and CBetFreq validation
- [ ] Statistical-unit and cluster-aware bootstrap validation
- [ ] Poetry lockfile and execution metadata persisted

**Deliverable:**

```bash
poker battle \
  --bot-a random \
  --bot-b tag \
  --seeds 10000
```

with a reproducible statistical report.

The statement "statistically valid" is only used after the sanity and precision tests have been passed.

---

### Phase 1: Baselines & Measurement (Weeks 3–4)

Phase 1 intentionally contains **no MCTS**. The purpose is to establish transparent baseline agents and verify that the benchmark infrastructure produces interpretable measurements before search complexity is introduced.

- [ ] Develop a random baseline agent
- [ ] Develop a deterministic hand-strength baseline
- [ ] Implement exact equity computation where feasible
- [ ] Implement pot-odds decisions
- [ ] Define and version the equity/range representation
- [ ] Full benchmark matrix against calibrated static profiles
- [ ] Measure performance against the baseline bet abstraction
- [ ] Run the predefined action-abstraction sensitivity checks where required
- [ ] Calibrate parametric opponents in the frozen calibration environment
- [ ] Verify reporting of hands, runs, profiles, statistical unit, seeds, and configuration hashes

The Phase-1 baseline is deliberately simple. Its purpose is measurement validation, not strategic strength.

---

### Phase 2: Search, Opponent Modeling & Game Theory (Weeks 5–8)

- [ ] Develop an imperfect-information MCTS agent (IS-MCTS or another explicitly specified imperfect-information method)
- 100 simulations per decision as the initial engineering baseline
- extension to 1,000 in later experiments if justified
- rollout policy: hand strength + pot odds
- UCB formula:

$$
UCB = Q(s,a) + c \sqrt{ \frac{\ln(N(s))}{N(s,a)} }
$$

with $c=1.0$.

- [ ] Define search depth as number of actions
- [ ] Ablation: Depth 1 / 2 / 4 / 8
- [ ] Enforce the hidden-information safety boundary during search
- [ ] Record search budget and timeout behaviour
- [ ] Bayesian online learner
- [ ] Estimation of VPIP/PFR
- [ ] Estimation of further observable frequencies
- [ ] Uncertainty model
- [ ] Adaptation horizons
- [ ] Exploitative agent
- [ ] Define dynamic preflop ranges through a versioned range representation
- [ ] Dynamic postflop strategy
- [ ] Integration of OpenSpiel
- [ ] Best‑response analysis in Leduc
- [ ] Validation of exploitability computation
- [ ] Analysis of correlation between Leduc exploitability and NLHE performance

Important:

Leduc exploitability is stored as a **separate metric** and is **not** misinterpreted as a direct NLHE exploitability score.

---

#### Phase-2 MCTS Rollout Semantics

MCTS is an **imperfect-information** method. A perfect-information search over the hidden true state is invalid because it leaks information unavailable to the agent. The exact search method (preferably Information Set MCTS) is versioned.

The rollout policy is deterministic and based on:

1. equity against a defined versioned range;
2. pot odds;
3. a heuristic payoff estimate for each legal action.

For a simplified call model, with equity $E$, pot before calling $P$, and additional call cost $C$: $\widehat{Payoff}(call)=E(P+C)-C$. Here $\widehat{Payoff}$ is a **heuristic rollout payoff estimate**, not a game-theoretic EV. Bet and raise actions require their own response model. If such a model is incomplete, the rollout remains explicitly labelled heuristic.

---

### Phase 3: Meta‑Optimisation & Pareto Analysis (Weeks 9–12)

- [ ] Train a simple reinforcement learning agent, e.g., NFSP
- [ ] Train against parametric development population
- [ ] Hyperparameter selection on validation population
- [ ] Final evaluation on unseen holdout population
- [ ] Ablation of opponent distribution
- [ ] Ablation of bet abstraction
- [ ] Derive three agent archetypes:

#### Titan

Minimises game‑theoretic exploitability in the Leduc environment.

Its performance in NLHE is measured as a transfer result.

A possible later variant can be trained in a reduced NLHE abstraction with CFR. In that case, an approximate exploitability estimate is explicitly labelled as **estimate** and not as exact exploitability.

#### Predator

Maximises population performance against clearly defined opponent populations or difficult population sub‑regions.

For example, Predator can be optimised for a pre‑defined high VPIP range.

The 80% percentile of the VPIP distribution can be used as a **concrete experiment**, but is not assumed as a general definition of "difficult".

#### Adaptive

Combines a stable baseline strategy with opponent modeling and switches between different strategic modes depending on the posterior certainty.

---

### 10.5 Pareto Analysis

The Pareto analysis uses **commensurable NLHE objectives** as its primary optimisation view:

$$
Y(\pi) = \overline{EV}_{NLHE}(\pi) \quad \text{maximise}
$$
$$
Z(\pi) = R_{10}(\pi) \quad \text{maximise}
$$

The standard Pareto visualisation therefore uses:

- X-axis: $\overline{EV}_{NLHE}$
- Y-axis: $R_{10}$

Leduc exploitability $X_{Leduc}$ remains a separate game-theoretic metric and is analysed in H5. It is not converted into BB/100 merely to share a plot with NLHE metrics.

Each agent is represented as **one aggregated point**. The population over which $\overline{EV}_{NLHE}$ and $R_{10}$ are computed is fixed before the experiment. A visualisation against individual $\theta$ -samples is treated as a separate sensitivity analysis and not mixed with the primary Pareto front.

The Pareto front is not assumed a priori, but determined empirically from the results.

---

## 20. First Technical Milestone

The first milestone is the development of the deterministic paired-match engine and its surrounding validation infrastructure. It is a concrete technical milestone for the project.

**No AI optimisation begins until the following conditions are met:**

- [ ] Repository and Poetry environment initialised
- [ ] Canonical configuration and SHA-256 hashing implemented
- [ ] Engine rules fully specified
- [ ] Bet/raise semantics formally defined, versioned, and covered by unit tests
- [ ] Hand evaluation validated
- [ ] Pot and stack computation validated
- [ ] Pre-defined deal schedule and RNG hierarchy reproducible
- [ ] Position compensation implemented
- [ ] Random‑vs‑random test passed
- [ ] Symmetry test passed
- [ ] Determinism test passed
- [ ] Engine regression tests passed
- [ ] Hidden-information safety test passed
- [ ] Illegal-action handling validated
- [ ] Agent budget/timeout policy validated
- [ ] Policy calibration tests passed
- [ ] Statistical evaluation validated
- [ ] Power and precision planning implemented
- [ ] Experiment configuration fully persisted
- [ ] Replay of any match is possible

Side‑pot and multiplayer tests will only become mandatory engine gates with the later 6‑max extension.

Only when these conditions are met and reproducible results for identical and random bots are available does the implementation of more complex AI algorithms begin.

Thus, the project consistently adheres to the principle:

> **Evaluation Framework first, Bot second.**

---

*End of Document*
