from decimal import Decimal

import pytest

from pokerbench.engine.betting_rules import (
    BettingRules,
    BettingRulesVersion,
)
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)


@pytest.fixture
def rules() -> BettingRules:
    return BettingRules.from_config(BASELINE_GAME_CONFIG)


def test_baseline_rules_version(rules: BettingRules) -> None:
    assert rules.version is BettingRulesVersion.BASELINE_V1


def test_baseline_rules_are_derived_from_config(
    rules: BettingRules,
) -> None:
    assert rules.config == BASELINE_GAME_CONFIG


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action.BET_33, Decimal("0.33")),
        (Action.BET_66, Decimal("0.66")),
        (Action.BET_100, Decimal("1")),
        (Action.BET_200, Decimal("2")),
    ],
)
def test_multiplier_mapping(
    rules: BettingRules,
    action: Action,
    expected: Decimal,
) -> None:
    assert rules.multiplier_for(action) == expected


@pytest.mark.parametrize(
    "action",
    [
        Action.FOLD,
        Action.CHECK,
        Action.CALL,
        Action.ALL_IN,
    ],
)
def test_non_bet_actions_have_no_multiplier(
    rules: BettingRules,
    action: Action,
) -> None:
    assert rules.multiplier_for(action) is None


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action.BET_33, 99),
        (Action.BET_66, 198),
        (Action.BET_100, 300),
        (Action.BET_200, 600),
    ],
)
def test_bet_amount_uses_street_start_pot(
    rules: BettingRules,
    action: Action,
    expected: int,
) -> None:
    assert rules.bet_amount(
        action=action,
        pot_at_street_start=300,
        stack=1000,
    ) == expected


def test_bet_amount_is_stack_limited(
    rules: BettingRules,
) -> None:
    assert rules.bet_amount(
        action=Action.BET_200,
        pot_at_street_start=300,
        stack=250,
    ) == 250


def test_bet_amount_returns_none_for_non_bet_action(
    rules: BettingRules,
) -> None:
    assert rules.bet_amount(
        action=Action.CALL,
        pot_at_street_start=300,
        stack=1000,
    ) is None


def test_raise_to_amount_uses_opponent_bet_and_street_pot(
    rules: BettingRules,
) -> None:
    assert rules.raise_to_amount(
        action=Action.BET_200,
        pot_at_street_start=300,
        opponent_bet=100,
        current_bet=50,
        stack=1000,
    ) == 700


def test_raise_to_amount_is_stack_limited(
    rules: BettingRules,
) -> None:
    assert rules.raise_to_amount(
        action=Action.BET_200,
        pot_at_street_start=300,
        opponent_bet=100,
        current_bet=50,
        stack=250,
    ) == 300


def test_minimum_bet_is_one_big_blind(
    rules: BettingRules,
) -> None:
    assert rules.min_bet() == 100


def test_first_minimum_raise_increment_is_one_big_blind(
    rules: BettingRules,
) -> None:
    assert rules.min_raise_increment(0) == 100


def test_subsequent_minimum_raise_increment_matches_last_raise(
    rules: BettingRules,
) -> None:
    assert rules.min_raise_increment(300) == 300


@pytest.mark.parametrize(
    "pot",
    [-1],
)
def test_negative_pot_is_rejected(
    rules: BettingRules,
    pot: int,
) -> None:
    with pytest.raises(ValueError):
        rules.bet_amount(
            action=Action.BET_33,
            pot_at_street_start=pot,
            stack=1000,
        )


def test_negative_stack_is_rejected(
    rules: BettingRules,
) -> None:
    with pytest.raises(ValueError):
        rules.bet_amount(
            action=Action.BET_33,
            pot_at_street_start=300,
            stack=-1,
        )