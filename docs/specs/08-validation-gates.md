# Validation Pyramid and Phase-0 Gates

## 6. Test Pyramid: Game‑Theoretic Validation vs. Population Performance

The evaluation is divided into three levels:

| Level | Environment | Purpose | Reference Implementation |
| :--- | :--- | :--- | :--- |
| **Unit / Proof** | **Leduc Hold'em** | Testing game‑theoretic computations | OpenSpiel / Best Response |
| **Integration** | **Limit Hold'em** | Validation of hand evaluation, game rules, and simple strategies | OpenSpiel / own engine |
| **Production** | **Abstracted NLHE** | Main agent benchmarking | Own Python engine |

These levels have **different scientific purposes**.

Leduc is not a direct simulation of NLHE, but a controllable environment in which best‑response and exploitability computations can be verified.

Limit Hold'em serves as an intermediate step for validating core game logic.

Abstracted NLHE is the actual production environment for population performance and robustness.

The later analysis explicitly examines whether properties from the smaller games correlate with NLHE performance.

---

## 9. Validation & Sanity Gates

Before complex AI methods are implemented, the engine must pass a series of tests.

---

### 9.1 Unit Tests: Engine

At least:

- Card deck
- Shuffle
- Hand ranking
- Board development
- Pot calculation
- Fold
- Check
- Call
- Bet
- Raise
- All‑in
- Minimum raise
- Position rotation
- Showdown
- Split pot
- Stack reset
- Illegal-action rejection
- Deterministic abstract-action-to-amount mapping

These tests verify only the correct implementation of the game mechanics.

---

### 9.2 Symmetry Test

Two completely identical bots must on average have no systematic advantage: $EV(A,A) \approx 0$

The experiment must in particular verify the position compensation and the RNG mechanics.

---

### 9.3 Random‑vs‑Random Test

Two identical random bots are played against each other over a large number of paired deals.

The result should be statistically compatible with zero and practically close to zero under the pre-defined symmetry tolerance $\epsilon_{sym}=0.5$ BB/100. The 95% CI of the paired difference must be fully contained in $[-0.5,+0.5]$ BB/100. A CI containing zero is a sanity check, not proof of engine correctness. Repeated seeds must also show no systematic positional or RNG bias.

The test is interpreted as a sanity test, not as proof of the general correctness of the engine.

---

### 9.4 Determinism Test

The same run:

```bash
poker run --seed 12345
```

must produce exactly the same sequence and the same result given identical software and configuration versions.

---

### 9.5 Regression Tests: Engine and Rule Tests

All known error cases are stored as fixed regression cases.

The regression tests include at least:

- defined preflop all‑in situations with **pre‑computed expected equity** and fixed card/deck definition
- defined river situations with known pot and stack outcomes
- split pot with identical hands
- correct handling of side pots as a separate future multiplayer test (this is prepared for the later 6‑max extension but is **not** part of the HU Phase‑0 gate; see 9.8)
- correct minimum‑raise rule
- correct stack limiting
- correct pot distribution
- correct position rotation

For fixed equity test cases, no general statements such as "AA vs 72o must deliver 80+% EV" are used.

Instead, **fully specified test cases** with a known reference value are used.

A distinction is explicitly made between **equity**, **realised profit**, and **EV**.

---

### 9.6 Strategy and Policy Tests

Strategic expectations are separated from engine regression tests.

Examples:

- A calling station shall achieve defined FoldToCBet target values.
- A TAG profile shall lie within a tolerance of its VPIP/PFR target values.
- A policy with a defined action shall correctly pass that action to the engine.
- A hand‑strength policy shall select the expected action in a fully specified situation.

These tests verify the policy, not the underlying engine mechanics.

---

### 9.7 Policy Calibration Test

For each parametric profile, a calibration run is performed.

The observed frequencies are compared with the target parameters:

```text
target VPIP       vs. observed VPIP
target PFR        vs. observed PFR
target 3Bet       vs. observed 3Bet
target CBetFreq   vs. observed CBetFreq
target Aggression vs. observed Aggression
target FoldToCBet vs. observed FoldToCBet
```

A profile is considered calibrated only if all relevant deviations lie within the defined tolerances.

For `parametric-policy-v1`, Gate 6 additionally requires all five `static-profiles-v2` profiles to pass `policy-calibration-environment-v1` as defined in `06-policy-calibration.md` and `16-parametric-policy-v1.md`. The independent validation sample must not be used for fitting the policy gates.

---

### 9.8 Hidden-Information Safety Gate

The engine maintains two explicitly separate representations:

```text
TRUE_GAME_STATE
AGENT_OBSERVATION
```

For player $i$, only

$$o_i = O_i(s)$$

is supplied to the policy. The observation must not contain opponent hole cards, undealt cards, future board cards, hidden deck state, hidden opponent RNG state, or any other information unavailable to that player.

A dedicated Phase-0 test constructs states with identical legal observations but different hidden information and verifies that a deterministic policy produces the same action. This test is mandatory before MCTS or RL agents are benchmarked.

### 9.9 Phase‑0 Gate

Phase 0 is considered complete only when the following conditions are met:

- Engine unit tests passed
- Symmetry test passed
- Random‑vs‑random test passed
- Determinism test passed
- Regression tests passed
- Policy calibration tests passed
- Statistical evaluation validated
- Statistical unit explicitly represented in configuration
- Cluster-aware bootstrap/CI validated for adaptive trajectories
- Hidden-information safety gate passed
- Illegal-action behaviour validated
- Agent budget and timeout behaviour defined
- Experiment configuration fully persisted
- Replay of any match is possible

Side‑pot and 6‑max tests are **not part of the HU Phase‑0 gate**; they will be activated with the later multiplayer extension.

---
