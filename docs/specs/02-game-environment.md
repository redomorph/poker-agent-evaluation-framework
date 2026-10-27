# Game Environment and Betting Abstraction

## 2. Problem Definition & Game Selection

### 2.1 Focus

**Heads-Up No-Limit Texas Hold'em (HU NLHE) in a controlled cash-game benchmark environment.**

To make the game fully reproducible, the benchmark environment is explicitly defined:

- 2 players
- Heads-Up
- No-Limit Texas Hold'em
- 100 BB starting stack
- Small Blind = 0.5 BB
- Big Blind = 1 BB
- no antes
- no rake in the scientific baseline benchmark
- stack reset to 100 BB after each hand
- no rebuys within a hand
- deterministic card order per experiment seed
- discrete, explicitly defined bet-sizing space
- identical game rules for all tested agents

This configuration initially does **not attempt to simulate a real poker room**, but defines a controlled research environment.

A later extension with rake, different stack sizes, antes, or 6-max will be treated as a separate experiment and not mixed with the baseline results.

### 2.2 Justification

- Two players → zero-sum structure and clear definition of net chip EV.
- No multiplayer dynamics such as team formation or collusion.
- Good suitability for later extensions.
- High strategic complexity due to hidden information.
- Bet-sizing and information sets present a relevant algorithmic challenge.
- The game definition can be fully controlled and reproduced.

---

## 5. The Central Technical Hurdle: Bet‑Sizing Abstraction

The full no‑limit bet space is continuous. An exact enumeration of all possible bet sizes leads to an extremely large action set.

Therefore, a discrete bet‑sizing abstraction is introduced for the benchmark.

### 5.1 Baseline Abstraction

```python
BET_ABSTRACTION = [0.33, 0.66, 1.0, 2.0]

ALLOWED_ACTIONS = [ FOLD, CHECK, CALL, BET_33, BET_66, BET_100, BET_200, ALL_IN, ]
```

The numbers represent standardised pot multipliers for bets, and – after a separate raise definition – for raises.

### 5.2 Exact Semantics

The abstraction must unambiguously define for every situation:

- Reference pot size
- Computation of a bet
- Computation of a raise
- Minimum bet
- Minimum raise
- Handling of all‑in
- Rounding
- Stack limiting
- Behaviour with short stacks

In particular, `BET_100` must not be interpreted differently depending on whether a bet has already occurred previously.

The concrete formula for raises is documented as part of the game definition. Before final benchmarking, the exact function must be immutable and versioned. The abstract action label is context-dependent: in an unopened betting situation, `BET_k` maps to a bet of size `kP`; in a raise situation, the same abstract label maps to a deterministic raise-to amount defined by the versioned function below. It is therefore one context-dependent abstract command, not two different action identifiers. For example, with reference pot

$$
Amount_{bet}=\min(kP,S)\,,\qquad Amount_{raise}=f(P,B,k,S)\,.
$$

Here, $P$ is the reference pot size, $B$ is the opponent wager, $k$ is the abstract multiplier, and $S$ is the maximum legal amount. The exact function $f$ must be specified. Minimum bet, minimum raise, incomplete all-in raises, action reopening, rounding, stack caps, and short-stack behaviour are part of the game configuration.

For example, a raise can be defined as a function of current pot, opponent's bet, desired multiplier, and stack limit:

$$
RaiseAmount = f(Pot, OpponentBet, k, S)
$$

The implementation must ensure that each abstract action identifier deterministically generates exactly one legal bet amount.

`ALL_IN` maps to the player's entire remaining stack and is treated as a bet if no wager exists on the current street, or as a raise-to the all-in amount otherwise. It is always accepted if the amount is at least the minimum legal bet/raise; otherwise it is treated as an illegal action.

### 5.3 Consequences

Thus, the agent is strictly playing an **abstracted version of NLHE**.

The evaluation therefore initially applies only to this defined abstract game environment.

A later transfer to more realistic bet sizes or human players will be treated as a separate experiment.

Additionally, the sensitivity to the abstraction will be investigated:

- smaller action sets
- baseline action set
- expanded action set

This allows empirical testing of how strongly the results depend on the chosen abstraction.

---
