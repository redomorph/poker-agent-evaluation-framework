import hashlib
import json
from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any


class GameType(StrEnum):
    NLHE = "no_limit_texas_holdem"


class Action(StrEnum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET_33 = "bet_33"
    BET_66 = "bet_66"
    BET_100 = "bet_100"
    BET_200 = "bet_200"
    ALL_IN = "all_in"


@dataclass(frozen=True)
class GameConfig:
    game_type: GameType
    player_count: int
    starting_stack_bb: Decimal
    small_blind_bb: Decimal
    big_blind_bb: Decimal
    ante_bb: Decimal
    rake: Decimal
    stack_reset_bb: Decimal
    bet_abstraction: tuple[Decimal, ...]
    allowed_actions: tuple[Action, ...]
    abstraction_version: str


BASELINE_GAME_CONFIG = GameConfig(
    game_type=GameType.NLHE,
    player_count=2,
    starting_stack_bb=Decimal("100"),
    small_blind_bb=Decimal("0.5"),
    big_blind_bb=Decimal("1"),
    ante_bb=Decimal("0"),
    rake=Decimal("0"),
    stack_reset_bb=Decimal("100"),
    bet_abstraction=(
        Decimal("0.33"),
        Decimal("0.66"),
        Decimal("1.0"),
        Decimal("2.0"),
    ),
    allowed_actions=(
        Action.FOLD,
        Action.CHECK,
        Action.CALL,
        Action.BET_33,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
        Action.ALL_IN,
    ),
    abstraction_version="baseline-v1",
)


def _normalize_decimal(value: Decimal) -> str:
    normalized = value.normalize()

    if normalized == Decimal("0"):
        return "0"

    return format(normalized, "f")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _normalize_decimal(value)

    if isinstance(value, StrEnum):
        return value.value

    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]

    if isinstance(value, list):
        return [_normalize_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _normalize_value(item)
            for key, item in value.items()
        }

    return value


def canonical_config_dict(config: GameConfig) -> dict[str, Any]:
    return _normalize_value(asdict(config))


def canonical_config_json(config: GameConfig) -> str:
    return json.dumps(
        canonical_config_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def config_hash(config: GameConfig) -> str:
    canonical_json = canonical_config_json(config)

    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()