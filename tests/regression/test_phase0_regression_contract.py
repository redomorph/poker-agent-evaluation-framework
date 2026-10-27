from __future__ import annotations

from fractions import Fraction

from pokerbench.engine.config import Action, BASELINE_GAME_CONFIG
from pokerbench.engine.hand_evaluator import HandEvaluator
from pokerbench.engine.state import GameState, Street


# Card encoding is rank + 13 * suit, with ranks 2..A mapped to 0..12.
# These fixtures are intentionally explicit rather than generated from the
# production deal schedule: they are frozen regression references.
HERO_AA: tuple[int, int] = (12, 25)  # AC AD
VILLAIN_KK: tuple[int, int] = (11, 24)  # KC KD

# A finite, explicitly frozen runout distribution for the preflop all-in
# reference.  The expected outcomes were fixed before the test calculation:
# hero win, villain win, hero win, tie.
PRECOMPUTED_ALL_IN_RUNOUTS: tuple[tuple[int, int, int, int, int], ...] = (
    (0, 18, 33, 48, 1),   # 2C 7D 9H JS 3C -> AA wins
    (37, 0, 18, 33, 48),  # KH 2C 7D 9H JS -> KK wins
    (38, 0, 18, 33, 48),  # AH 2C 7D 9H JS -> AA wins
    (0, 14, 28, 42, 4),   # 2C 3D 4H 5S 6C -> board straight, tie
)
PRECOMPUTED_ALL_IN_OUTCOMES: tuple[int, ...] = (1, -1, 1, 0)
PRECOMPUTED_ALL_IN_EQUITY: Fraction = Fraction(5, 8)


def _showdown_outcome(
    hero_hole: tuple[int, int],
    villain_hole: tuple[int, int],
    board: tuple[int, int, int, int, int],
) -> int:
    hero_best = HandEvaluator.best_hand((*hero_hole, *board))
    villain_best = HandEvaluator.best_hand((*villain_hole, *board))
    hero_value = HandEvaluator.evaluate(hero_best)
    villain_value = HandEvaluator.evaluate(villain_best)

    if hero_value > villain_value:
        return 1
    if hero_value < villain_value:
        return -1
    return 0


def test_preflop_all_in_has_precomputed_reference_equity() -> None:
    """Validate a frozen preflop all-in equity reference independently of EV."""
    outcomes = tuple(
        _showdown_outcome(
            HERO_AA,
            VILLAIN_KK,
            board,
        )
        for board in PRECOMPUTED_ALL_IN_RUNOUTS
    )

    assert outcomes == PRECOMPUTED_ALL_IN_OUTCOMES

    wins = outcomes.count(1)
    ties = outcomes.count(0)
    reference_equity = Fraction(
        2 * wins + ties,
        2 * len(outcomes),
    )

    assert reference_equity == PRECOMPUTED_ALL_IN_EQUITY


def test_preflop_all_in_realized_profit_is_not_reference_equity() -> None:
    """Keep realised profit distinct from the precomputed equity fixture."""
    state = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(HERO_AA, VILLAIN_KK),
        board=PRECOMPUTED_ALL_IN_RUNOUTS[0],
        dealer=0,
    )

    state = state.apply_action(Action.ALL_IN)
    state = state.apply_action(Action.CALL)

    assert state.is_terminal
    assert state.terminal_profit == (10_000, -10_000)
    assert PRECOMPUTED_ALL_IN_EQUITY == Fraction(5, 8)


def test_river_regression_has_known_pot_stack_and_profit_outcome() -> None:
    """Freeze a complete river call with known pot, stacks, and net profit."""
    state = GameState.new_hand(
        config=BASELINE_GAME_CONFIG,
        hole_cards=(HERO_AA, VILLAIN_KK),
        board=PRECOMPUTED_ALL_IN_RUNOUTS[0],
        dealer=0,
    )

    actions = (
        Action.CALL,
        Action.CHECK,
        Action.CHECK,
        Action.BET_100,
        Action.CALL,
        Action.CHECK,
        Action.CHECK,
        Action.CHECK,
        Action.BET_100,
        Action.CALL,
    )

    for action in actions:
        state = state.apply_action(action)

    assert state.street == Street.RIVER
    assert state.is_terminal
    assert state.pot == 1_800
    assert state.stacks == (9_100, 9_100)
    assert state.terminal_profit == (900, -900)
