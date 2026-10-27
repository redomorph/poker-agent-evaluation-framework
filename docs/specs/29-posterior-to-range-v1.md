# Posterior-to-Range v1

## Status

This specification freezes the first deterministic mapping from the Bayesian opponent model to an opponent hidden-card range for the Phase-2 adaptive baseline.

```text
protocol_id      = phase2-posterior-to-range-v1
protocol_version = posterior-to-range-v1
status           = frozen
scope            = posterior-to-preflop-range-only
```

It depends on:

```text
bayesian-opponent-model-v1
adaptive-learning-protocol-v1
public-action-trace-v1
uniform-random-equity-v1
uniform-random-range-v1
```

The machine-readable protocol is:

```text
configs/experiments/phase2-posterior-to-range-v1.json
```

The frozen preflop strength reference is:

```text
configs/experiments/preflop-range-strength-reference-v1.json
```

Its payload SHA-256 is:

```text
cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e
```

## 1. Purpose and non-goals

The v1 mechanism makes one deliberately narrow modeling claim:

> observable preflop participation and raising frequencies may be translated into a deterministic width of a preflop opponent range.

It does **not** claim that the six behavioral statistics uniquely identify an opponent's complete hidden-card strategy.

The following remain out of scope:

- posterior-to-hero-action mapping;
- action-conditioned postflop range narrowing;
- posterior-uncertainty action adjustment;
- H4 experimental design;
- MCTS semantics.

## 2. Posterior statistic

Range construction uses posterior means only. For metric $m$,

$$
\widehat q_m = \frac{\alpha_m}{\alpha_m+\beta_m}.
$$

The posterior itself is never sampled. Thompson sampling is not used. Range construction consumes zero RNG draws.

For serialization and range sizing, posterior means are converted to integer parts-per-million using deterministic nearest-half-up rounding:

$$
q_{m,ppm}
=
\left\lfloor
\frac{10^6\alpha_m + (\alpha_m+\beta_m)/2}
{\alpha_m+\beta_m}
\right\rfloor
$$

with the integer implementation defined by the protocol validator.

## 3. Range-consistency projection

The Bayesian posteriors remain unchanged. A separate deterministic projection is used only to enforce nested preflop range widths:

$$
q^*_{VPIP}=\widehat q_{VPIP},
$$

$$
q^*_{PFR}=\min(\widehat q_{PFR},q^*_{VPIP}),
$$

$$
q^*_{3Bet}=\min(\widehat q_{3Bet},q^*_{PFR}).
$$

Thus:

$$
R_{3Bet}\subseteq R_{PFR}\subseteq R_{VPIP}.
$$

This is a range-consistency transform, not a Bayesian posterior correction.

## 4. Public preflop signal

Only public preflop actions are used. For one opponent identity, the strongest v1 signal is:

| Public opponent line | v1 signal | Width source |
|---|---|---|
| no informative voluntary action | `uninformative` | 100% |
| passive voluntary entry | `passive_entry` | effective VPIP |
| first voluntary wager increase | `first_raise` | effective PFR |
| wager increase after exactly one prior public wager increase | `three_bet` | effective 3Bet |

`CALL` and a non-increasing `ALL_IN` may establish passive entry. `CHECK` and `FOLD` do not establish a voluntary-entry range.

A 4-bet or higher wager increase does not create a new v1 statistic. The last already-established v1 signal is retained. Therefore a player who first raised and later 4-bet remains on the PFR-derived v1 range rather than receiving an invented 4-bet range.

Non-preflop public events do not modify the preflop signal.

## 5. Canonical preflop strength reference

### 5.1 Hand classes

The 169 standard rank/suitedness classes are enumerated deterministically:

1. iterate the high rank from Ace down to Deuce;
2. emit the pair for that rank;
3. for each lower rank from highest to lowest, emit the suited class and then the offsuit class.

Examples begin:

```text
AA, AKs, AKo, AQs, AQo, ...
```

A class index is its zero-based index in this enumeration.

### 5.2 Representative cards

With card encoding `rank + 13 * suit`, suits ordered Clubs, Diamonds, Hearts, Spades:

- pair `RR`: Clubs + Diamonds;
- suited `HLs`: both Clubs;
- offsuit `HLo`: high card Clubs + low card Diamonds.

### 5.3 Frozen equity score

Each class representative is scored once using `uniform-random-equity-v1` at preflop with:

```text
reference_master_seed = 7_200_001
bot_id                = 0
hand_id               = canonical class index
decision_index        = 0
samples               = 10_000
```

The stored score is the resulting integer equity ppm.

The reference was frozen before adaptive evaluation. It is a computational ordering reference, not an empirical population estimate.

Because preflop `uniform-random-equity-v1` uses Monte Carlo at this state size, near-adjacent class ordering contains deterministic Monte-Carlo approximation noise. A later higher-precision or exact reference would require a new version.

### 5.4 Class and combination order

Classes are sorted by:

```text
1. equity_ppm descending
2. class_id ASCII ascending
```

Each class expands to physical combinations:

- pair: 6 combinations;
- suited non-pair: 4 combinations;
- offsuit non-pair: 12 combinations.

Within a class, physical combinations are sorted by the canonical tuple:

```text
(min(card_a, card_b), max(card_a, card_b))
```

ascending.

Expansion of all 169 classes must produce exactly 1,326 unique two-card combinations.

## 6. Quantile range construction

For effective width $q_{ppm}$ and $K=1326$ physical combinations, select:

$$
N = \max\left(1,\left\lceil\frac{q_{ppm}K}{10^6}\right\rceil\right).
$$

The range is the first $N$ combinations in the frozen full strength order.

Examples are therefore exact in **physical combination space**, not in 169-class space. A quantile boundary may split a hand class.

## 7. Blockers

Known hero and board cards are applied **after** the quantile range is selected:

$$
R_{legal}=
\{c\in R: c\cap K_{known}=\varnothing\}.
$$

The framework must not recompute the top-$q$ fraction of the remaining deck and must not backfill blocked combinations with weaker hands.

Every surviving selected combination has equal probability:

$$
P(c\mid R_{legal})=\frac{1}{|R_{legal}|}.
$$

If all selected combinations are blocked, v1 raises an explicit range-construction error. Silent widening/backfill is prohibited.

## 8. Postflop treatment

The last inferred preflop range is carried forward after the flop. Newly visible board cards are removed as blockers.

In v1:

```text
postflop action-conditioned narrowing = false
```

CBet, FoldToCBet, and postflop aggressive-share posteriors do not alter the hidden-card distribution.

This is intentional. Those metrics describe behavior but do not uniquely determine a hidden-card likelihood model. A future action-conditioned range model requires a separate versioned specification.

## 9. Metrics unused for v1 card range

The following posterior components are persisted and learned but do not change v1 card-range width or membership:

```text
CBet
FoldToCBet
postflop aggressive share
```

They may be used by a later posterior-to-action policy after that policy is separately frozen.

## 10. Range identity and replay

A range snapshot binds at least:

```text
posterior-to-range version
preflop signal
effective width ppm
selected combination count
ordered selected combinations
Bayesian model state hash
preflop strength reference hash
```

The canonical representation receives a SHA-256 range hash.

Two opponent models that imply identical selected combinations but have different complete posterior states may therefore have different range snapshot hashes. This is intentional provenance binding; range membership equality and model-state identity are separate facts.

## 11. RNG

```text
posterior sampling        = false
range-construction draws  = 0
```

Reference generation uses its separately frozen equity seed only when the reference artifact is produced. Runtime range construction performs no random sampling.

## 12. Validation gate

Before an adaptive policy uses this mechanism, tests must verify at minimum:

- canonical protocol and reference hashes;
- all 169 hand classes;
- expansion to exactly 1,326 unique physical combinations;
- frozen reference score anchors;
- exact nesting projection;
- exact public-line classification;
- integer-ceiling quantile boundaries;
- blocker filtering without backfill;
- postflop behavioral metrics do not alter v1 card-range membership;
- deterministic/model-bound range hashes;
- non-preflop action immunity;
- empty blocked range rejection;
- semantic tamper rejection even when an outer JSON hash is recomputed.

## 13. Next freeze

After this specification passes its validation gate, the next scientific dependency is **range-conditioned equity and posterior-to-action-policy v1**.

That future specification must define exactly how the inferred range modifies the otherwise identical non-adaptive hand-strength baseline and how CBet/FoldToCBet/aggressive-share posteriors may alter hero action thresholds. H4 sample size and confirmatory inference remain out of scope until the adaptive policy mechanism itself is validated.
