import json
from dataclasses import replace
from decimal import Decimal

import pytest

from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
    GameType,
    canonical_config_dict,
    canonical_config_json,
    config_hash,
)


def test_baseline_game_type():
    assert BASELINE_GAME_CONFIG.game_type is GameType.NLHE


def test_baseline_player_count():
    assert BASELINE_GAME_CONFIG.player_count == 2


def test_baseline_starting_stack():
    assert BASELINE_GAME_CONFIG.starting_stack_bb == Decimal("100")


def test_baseline_blinds():
    assert BASELINE_GAME_CONFIG.small_blind_bb == Decimal("0.5")
    assert BASELINE_GAME_CONFIG.big_blind_bb == Decimal("1")


def test_baseline_no_ante():
    assert BASELINE_GAME_CONFIG.ante_bb == Decimal("0")


def test_baseline_no_rake():
    assert BASELINE_GAME_CONFIG.rake == Decimal("0")


def test_baseline_stack_reset():
    assert BASELINE_GAME_CONFIG.stack_reset_bb == Decimal("100")


def test_baseline_bet_abstraction():
    assert BASELINE_GAME_CONFIG.bet_abstraction == (
        Decimal("0.33"),
        Decimal("0.66"),
        Decimal("1.0"),
        Decimal("2.0"),
    )


def test_baseline_allowed_actions():
    assert BASELINE_GAME_CONFIG.allowed_actions == (
        Action.FOLD,
        Action.CHECK,
        Action.CALL,
        Action.BET_33,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
        Action.ALL_IN,
    )


def test_baseline_config_is_immutable():
    with pytest.raises(AttributeError):
        BASELINE_GAME_CONFIG.player_count = 3  # type: ignore


def test_canonical_config_contains_expected_values():
    canonical = canonical_config_dict(BASELINE_GAME_CONFIG)

    assert canonical["game_type"] == "no_limit_texas_holdem"
    assert canonical["player_count"] == 2
    assert canonical["starting_stack_bb"] == "100"
    assert canonical["small_blind_bb"] == "0.5"
    assert canonical["big_blind_bb"] == "1"


def test_canonical_json_is_valid_json():
    parsed = json.loads(
        canonical_config_json(BASELINE_GAME_CONFIG)
    )

    assert parsed["player_count"] == 2


def test_canonical_json_has_sorted_keys():
    parsed = json.loads(
        canonical_config_json(BASELINE_GAME_CONFIG)
    )

    assert list(parsed) == sorted(parsed)


def test_same_config_has_same_hash():
    assert config_hash(BASELINE_GAME_CONFIG) == config_hash(
        BASELINE_GAME_CONFIG
    )


def test_hash_is_sha256_hex():
    result = config_hash(BASELINE_GAME_CONFIG)

    assert len(result) == 64
    assert all(
        character in "0123456789abcdef"
        for character in result
    )


def test_changed_config_changes_hash():
    modified = replace(
        BASELINE_GAME_CONFIG,
        starting_stack_bb=Decimal("200"),
    )

    assert config_hash(BASELINE_GAME_CONFIG) != config_hash(modified)


def test_changed_bet_abstraction_changes_hash():
    modified = replace(
        BASELINE_GAME_CONFIG,
        bet_abstraction=(
            Decimal("0.25"),
            Decimal("0.5"),
            Decimal("1.0"),
            Decimal("2.0"),
        ),
    )

    assert config_hash(BASELINE_GAME_CONFIG) != config_hash(modified)


def test_equivalent_decimal_representations_have_same_hash():
    modified = replace(
        BASELINE_GAME_CONFIG,
        starting_stack_bb=Decimal("100.00"),
    )

    assert config_hash(BASELINE_GAME_CONFIG) == config_hash(modified)