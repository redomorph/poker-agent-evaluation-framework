"""
Replay inspection and filtering.

The CLI layer operates entirely on persisted MatchRecord objects and the
validated deterministic replay subsystem. Poker rules are never reimplemented
here.

Filter language v1 intentionally supports conjunctions only:

    hero_loss > 50 AND street == 'river'

Supported fields:

    hero_loss
    hero_profit
    street
    action
    pot
    max_pot

Supported numeric operators:

    >
    >=
    <
    <=
    ==
    !=

String fields support:

    ==
    !=

The implementation never calls eval().
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pokerbench.engine.config import (
    Action,
)
from pokerbench.engine.constants import (
    CHIP_UNIT,
)
from pokerbench.engine.state import (
    Street,
)
from pokerbench.replay import (
    MatchRecord,
    RecordedHand,
)


FILTER_VERSION = "inspect-filter-v1"

_VALID_STREETS = frozenset(
    {
        "preflop",
        "flop",
        "turn",
        "river",
        "showdown",
    }
)

_FILTER_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<field>[A-Za-z_][A-Za-z0-9_]*)
    \s*
    (?P<operator>==|!=|>=|<=|>|<)
    \s*
    (?P<value>.+?)
    \s*$
    """,
    flags=re.VERBOSE,
)

_AND_PATTERN = re.compile(
    r"\s+AND\s+",
    flags=re.IGNORECASE,
)

_OR_PATTERN = re.compile(
    r"\s+OR\s+",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class InspectOptions:
    """Frozen replay-inspection selection criteria."""

    hero_bot_id: int = 0

    street: str | None = None
    action: Action | None = None

    min_pot_bb: float | None = None

    filter_expression: str | None = None

    filter_version: str = (
        FILTER_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if self.hero_bot_id not in (
            0,
            1,
        ):
            raise ValueError(
                "hero_bot_id must be 0 or 1"
            )

        if self.street is not None:
            normalized = (
                self.street
                .strip()
                .lower()
            )

            if (
                normalized
                not in _VALID_STREETS
            ):
                raise ValueError(
                    "Unsupported street: "
                    f"{self.street}"
                )

            object.__setattr__(
                self,
                "street",
                normalized,
            )

        if self.min_pot_bb is not None:
            if (
                not isinstance(
                    self.min_pot_bb,
                    (int, float),
                )
                or isinstance(
                    self.min_pot_bb,
                    bool,
                )
            ):
                raise TypeError(
                    "min_pot_bb must be numeric"
                )

            numeric = float(
                self.min_pot_bb
            )

            if (
                not math.isfinite(
                    numeric
                )
                or numeric < 0.0
            ):
                raise ValueError(
                    "min_pot_bb must be finite "
                    "and non-negative"
                )

        if (
            self.filter_version
            != FILTER_VERSION
        ):
            raise ValueError(
                "Unsupported inspect-filter version"
            )


@dataclass(frozen=True)
class DecisionSnapshot:
    """One reconstructed decision before the chosen action is applied."""

    action_index: int

    actor_seat: int
    actor_bot_id: int

    street: str
    position: str

    pot_bb: float

    stacks_bb: tuple[
        float,
        float,
    ]

    current_bets_bb: tuple[
        float,
        float,
    ]

    full_board: tuple[int, ...]
    visible_board: tuple[int, ...]

    actor_hole_cards: tuple[
        int,
        int,
    ]

    opponent_hole_cards: tuple[
        int,
        int,
    ]

    action_history: tuple[
        Action,
        ...,
    ]

    legal_actions: tuple[
        Action,
        ...,
    ]

    chosen_action: Action


@dataclass(frozen=True)
class InspectedHand:
    """One persisted hand together with its reconstructed decisions."""

    record: RecordedHand

    decisions: tuple[
        DecisionSnapshot,
        ...,
    ]

    hero_seat: int

    hero_profit_bb: float | None

    max_pot_bb: float


@dataclass(frozen=True)
class _FilterClause:
    field: str
    operator: str
    raw_value: str


def inspect_match(
    match_record: MatchRecord,
    options: InspectOptions,
) -> tuple[
    InspectedHand,
    ...,
]:
    """
    Reconstruct, validate, and filter all hands from one MatchRecord.
    """
    expression_predicates = (
        _compile_filter_expression(
            options.filter_expression
        )
    )

    results: list[
        InspectedHand
    ] = []

    for hand in (
        match_record.hands
    ):
        inspected = (
            _inspect_hand(
                match_record,
                hand,
                hero_bot_id=(
                    options.hero_bot_id
                ),
            )
        )

        if (
            options.street
            is not None
            and not any(
                decision.street
                == options.street
                for decision
                in inspected.decisions
            )
        ):
            continue

        if (
            options.action
            is not None
            and not any(
                decision.chosen_action
                == options.action
                for decision
                in inspected.decisions
            )
        ):
            continue

        if (
            options.min_pot_bb
            is not None
            and inspected.max_pot_bb
            < options.min_pot_bb
        ):
            continue

        if not all(
            predicate(
                inspected
            )
            for predicate
            in expression_predicates
        ):
            continue

        results.append(
            inspected
        )

    return tuple(
        results
    )


def render_inspection_text(
    match_record: MatchRecord,
    hands: tuple[
        InspectedHand,
        ...,
    ],
    *,
    hero_bot_id: int,
) -> str:
    """Render complete replay inspection as human-readable text."""
    lines: list[str] = [
        f"Match: {match_record.match_id}",
        f"Master seed: {match_record.master_seed}",
        f"Game config hash: {match_record.game_config_hash}",
        f"Hero bot id: {hero_bot_id}",
        "",
    ]

    if not hands:
        lines.append(
            "No matching hands."
        )

        return "\n".join(
            lines
        )

    for inspected in hands:
        record = (
            inspected.record
        )

        replay = record.replay(
            match_record.game_config
        )

        initial = (
            replay.engine_replay
            .initial_state
        )

        opponent_seat = (
            1
            - inspected.hero_seat
        )

        lines.extend(
            [
                "=" * 78,
                (
                    f"Hand "
                    f"{record.hand_index}"
                ),
                (
                    "Termination: "
                    f"{record.termination.value}"
                ),
                (
                    "Hero seat: "
                    f"{inspected.hero_seat}"
                ),
                (
                    "Hero position: "
                    f"{_position_name(initial.dealer, inspected.hero_seat)}"
                ),
                (
                    "Hole cards: "
                    f"hero={initial.hole_cards[inspected.hero_seat]}, "
                    f"opponent={initial.hole_cards[opponent_seat]}"
                ),
                (
                    "Board: "
                    f"{initial.board}"
                ),
                (
                    "Resulting profit: "
                    f"{_format_optional_bb(inspected.hero_profit_bb)}"
                ),
                "",
            ]
        )

        for decision in (
            inspected.decisions
        ):
            lines.extend(
                [
                    (
                        f"Decision "
                        f"{decision.action_index}"
                    ),
                    (
                        "  Street: "
                        f"{decision.street}"
                    ),
                    (
                        "  Actor: "
                        f"bot {decision.actor_bot_id} "
                        f"(seat {decision.actor_seat})"
                    ),
                    (
                        "  Position: "
                        f"{decision.position}"
                    ),
                    (
                        "  Stacks: "
                        f"{decision.stacks_bb} BB"
                    ),
                    (
                        "  Pot: "
                        f"{decision.pot_bb:.2f} BB"
                    ),
                    (
                        "  Board: "
                        f"{decision.full_board}"
                    ),
                    (
                        "  Agent-visible board: "
                        f"{decision.visible_board}"
                    ),
                    (
                        "  Actor hole cards: "
                        f"{decision.actor_hole_cards}"
                    ),
                    (
                        "  Action history: "
                        f"{_actions_text(decision.action_history)}"
                    ),
                    (
                        "  Available actions: "
                        f"{_actions_text(decision.legal_actions)}"
                    ),
                    (
                        "  Chosen action: "
                        f"{decision.chosen_action.value}"
                    ),
                    (
                        "  Bot evaluation: N/A"
                    ),
                    (
                        "  Estimated hand strength: N/A"
                    ),
                    (
                        "  Estimated decision EV: N/A"
                    ),
                    "",
                ]
            )

        if (
            record.requested_action
            is not None
            and record.termination.value
            != "normal"
        ):
            lines.extend(
                [
                    "Exceptional termination:",
                    (
                        "  Requested action: "
                        f"{record.requested_action.value}"
                    ),
                    (
                        "  Legal actions: "
                        f"{_actions_text(record.legal_actions_at_termination)}"
                    ),
                    (
                        "  Failure message: "
                        f"{record.failure_message}"
                    ),
                    "",
                ]
            )

    return "\n".join(
        lines
    )


def render_inspection_html(
    match_record: MatchRecord,
    hands: tuple[
        InspectedHand,
        ...,
    ],
    *,
    hero_bot_id: int,
) -> str:
    """Render a minimal self-contained HTML replay viewer."""
    body: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Poker Replay Inspection</title>",
        "<style>",
        (
            "body{font-family:system-ui,sans-serif;"
            "max-width:1100px;margin:2rem auto;"
            "padding:0 1rem;line-height:1.45}"
        ),
        (
            "table{border-collapse:collapse;"
            "width:100%;margin:1rem 0}"
        ),
        (
            "th,td{border:1px solid #bbb;"
            "padding:.45rem;text-align:left;"
            "vertical-align:top}"
        ),
        (
            ".hand{margin:2rem 0;"
            "padding:1rem;border:1px solid #999}"
        ),
        (
            ".decision{margin:1rem 0;"
            "padding:1rem;border:1px solid #ccc}"
        ),
        "code{white-space:pre-wrap}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Poker Replay Inspection</h1>",
        (
            "<p><strong>Match:</strong> "
            f"{html.escape(match_record.match_id)}</p>"
        ),
        (
            "<p><strong>Master seed:</strong> "
            f"{match_record.master_seed}</p>"
        ),
        (
            "<p><strong>Hero bot id:</strong> "
            f"{hero_bot_id}</p>"
        ),
    ]

    if not hands:
        body.append(
            "<p>No matching hands.</p>"
        )

    for inspected in hands:
        record = (
            inspected.record
        )

        replay = record.replay(
            match_record.game_config
        )

        initial = (
            replay.engine_replay
            .initial_state
        )

        opponent_seat = (
            1
            - inspected.hero_seat
        )

        body.extend(
            [
                '<section class="hand">',
                (
                    "<h2>Hand "
                    f"{record.hand_index}</h2>"
                ),
                "<table>",
                (
                    "<tr><th>Termination</th><td>"
                    f"{html.escape(record.termination.value)}"
                    "</td></tr>"
                ),
                (
                    "<tr><th>Position</th><td>"
                    f"{html.escape(_position_name(initial.dealer, inspected.hero_seat))}"
                    "</td></tr>"
                ),
                (
                    "<tr><th>Hole cards</th><td>"
                    "Hero: "
                    f"{html.escape(str(initial.hole_cards[inspected.hero_seat]))}"
                    "<br>Opponent: "
                    f"{html.escape(str(initial.hole_cards[opponent_seat]))}"
                    "</td></tr>"
                ),
                (
                    "<tr><th>Board</th><td>"
                    f"{html.escape(str(initial.board))}"
                    "</td></tr>"
                ),
                (
                    "<tr><th>Resulting profit</th><td>"
                    f"{html.escape(_format_optional_bb(inspected.hero_profit_bb))}"
                    "</td></tr>"
                ),
                "</table>",
            ]
        )

        for decision in (
            inspected.decisions
        ):
            body.extend(
                [
                    '<div class="decision">',
                    (
                        "<h3>Decision "
                        f"{decision.action_index}</h3>"
                    ),
                    "<table>",
                    _html_row(
                        "Street",
                        decision.street,
                    ),
                    _html_row(
                        "Position",
                        decision.position,
                    ),
                    _html_row(
                        "Stacks",
                        (
                            f"{decision.stacks_bb} BB"
                        ),
                    ),
                    _html_row(
                        "Pot",
                        (
                            f"{decision.pot_bb:.2f} BB"
                        ),
                    ),
                    _html_row(
                        "Board",
                        str(
                            decision.full_board
                        ),
                    ),
                    _html_row(
                        "Agent-visible board",
                        str(
                            decision.visible_board
                        ),
                    ),
                    _html_row(
                        "Actor hole cards",
                        str(
                            decision.actor_hole_cards
                        ),
                    ),
                    _html_row(
                        "Action history",
                        _actions_text(
                            decision.action_history
                        ),
                    ),
                    _html_row(
                        "Available actions",
                        _actions_text(
                            decision.legal_actions
                        ),
                    ),
                    _html_row(
                        "Chosen action",
                        decision.chosen_action.value,
                    ),
                    _html_row(
                        "Bot evaluation",
                        "N/A",
                    ),
                    _html_row(
                        "Estimated hand strength",
                        "N/A",
                    ),
                    _html_row(
                        "Estimated decision EV",
                        "N/A",
                    ),
                    "</table>",
                    "</div>",
                ]
            )

        body.append(
            "</section>"
        )

    body.extend(
        [
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(
        body
    )


def write_inspection_html(
    path: str | Path,
    content: str,
) -> Path:
    """Write HTML inspection output atomically enough for local CLI use."""
    destination = Path(
        path
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        content,
        encoding="utf-8",
    )

    return destination


def _inspect_hand(
    match_record: MatchRecord,
    hand: RecordedHand,
    *,
    hero_bot_id: int,
) -> InspectedHand:
    replay_result = hand.replay(
        match_record.game_config
    )

    replay = (
        replay_result
        .engine_replay
    )

    hero_seat = (
        _seat_for_bot(
            hand,
            hero_bot_id,
        )
    )

    hero_profit: float | None

    if (
        hand.terminal_profit_bb
        is None
        or hero_seat is None
    ):
        hero_profit = None

    else:
        hero_profit = (
            hand.terminal_profit_bb[
                hero_seat
            ]
        )

    decisions: list[
        DecisionSnapshot
    ] = []

    for (
        action_index,
        action,
    ) in enumerate(
        hand.actions
    ):
        state = (
            replay.states[
                action_index
            ]
        )

        actor_seat = (
            state.current_player
        )

        actor_bot_id = (
            hand.seat0_bot_id
            if actor_seat == 0
            else hand.seat1_bot_id
        )

        observation = (
            state.observation(
                actor_seat
            )
        )

        decisions.append(
            DecisionSnapshot(
                action_index=(
                    action_index
                ),
                actor_seat=(
                    actor_seat
                ),
                actor_bot_id=(
                    actor_bot_id
                ),
                street=(
                    _street_name(
                        state.street
                    )
                ),
                position=(
                    _position_name(
                        state.dealer,
                        actor_seat,
                    )
                ),
                pot_bb=(
                    state.pot
                    / CHIP_UNIT
                ),
                stacks_bb=(
                    state.stacks[0]
                    / CHIP_UNIT,
                    state.stacks[1]
                    / CHIP_UNIT,
                ),
                current_bets_bb=(
                    state.current_bets[0]
                    / CHIP_UNIT,
                    state.current_bets[1]
                    / CHIP_UNIT,
                ),
                full_board=tuple(
                    state.board
                ),
                visible_board=tuple(
                    observation.board
                ),
                actor_hole_cards=(
                    observation.hole_cards
                ),
                opponent_hole_cards=(
                    state.hole_cards[
                        1 - actor_seat
                    ]
                ),
                action_history=tuple(
                    state.action_history
                ),
                legal_actions=tuple(
                    state.legal_actions()
                ),
                chosen_action=(
                    action
                ),
            )
        )

    max_pot_bb = max(
        (
            state.pot
            / CHIP_UNIT
            for state
            in replay.states
        ),
        default=0.0,
    )

    return InspectedHand(
        record=hand,
        decisions=tuple(
            decisions
        ),
        hero_seat=(
            hero_seat
            if hero_seat is not None
            else -1
        ),
        hero_profit_bb=(
            hero_profit
        ),
        max_pot_bb=(
            max_pot_bb
        ),
    )


def _compile_filter_expression(
    expression: str | None,
) -> tuple[
    Callable[
        [InspectedHand],
        bool,
    ],
    ...,
]:
    if expression is None:
        return ()

    normalized = (
        expression.strip()
    )

    if not normalized:
        return ()

    if _OR_PATTERN.search(
        normalized
    ):
        raise ValueError(
            "Inspect filter v1 supports AND "
            "conjunctions only"
        )

    clauses = (
        _AND_PATTERN.split(
            normalized
        )
    )

    predicates: list[
        Callable[
            [InspectedHand],
            bool,
        ]
    ] = []

    for raw_clause in clauses:
        match = (
            _FILTER_PATTERN.match(
                raw_clause
            )
        )

        if match is None:
            raise ValueError(
                "Invalid inspect filter clause: "
                f"{raw_clause}"
            )

        clause = _FilterClause(
            field=(
                match.group(
                    "field"
                )
                .lower()
            ),
            operator=(
                match.group(
                    "operator"
                )
            ),
            raw_value=(
                match.group(
                    "value"
                )
            ),
        )

        predicates.append(
            _compile_clause(
                clause
            )
        )

    return tuple(
        predicates
    )


def _compile_clause(
    clause: _FilterClause,
) -> Callable[
    [InspectedHand],
    bool,
]:
    if clause.field in (
        "street",
        "action",
    ):
        expected = (
            _parse_string_literal(
                clause.raw_value
            )
            .lower()
        )

        if clause.operator not in (
            "==",
            "!=",
        ):
            raise ValueError(
                "String filter fields support "
                "only == and !="
            )

        if (
            clause.field
            == "street"
        ):
            if (
                expected
                not in _VALID_STREETS
            ):
                raise ValueError(
                    "Unsupported street filter value: "
                    f"{expected}"
                )

            def street_predicate(
                inspected: InspectedHand,
            ) -> bool:
                present = any(
                    decision.street
                    == expected
                    for decision
                    in inspected.decisions
                )

                return (
                    present
                    if clause.operator
                    == "=="
                    else not present
                )

            return (
                street_predicate
            )

        valid_actions = {
            action.value
            for action
            in Action
        }

        if expected not in (
            valid_actions
        ):
            raise ValueError(
                "Unsupported action filter value: "
                f"{expected}"
            )

        def action_predicate(
            inspected: InspectedHand,
        ) -> bool:
            present = any(
                decision
                .chosen_action
                .value
                == expected
                for decision
                in inspected.decisions
            )

            return (
                present
                if clause.operator
                == "=="
                else not present
            )

        return (
            action_predicate
        )

    if clause.field in (
        "hero_loss",
        "hero_profit",
        "pot",
        "max_pot",
    ):
        expected_number = (
            _parse_numeric_literal(
                clause.raw_value
            )
        )

        def numeric_predicate(
            inspected: InspectedHand,
        ) -> bool:
            actual = (
                _numeric_field_value(
                    inspected,
                    clause.field,
                )
            )

            if actual is None:
                return False

            return _compare_numeric(
                actual,
                clause.operator,
                expected_number,
            )

        return (
            numeric_predicate
        )

    raise ValueError(
        "Unsupported filter field: "
        f"{clause.field}"
    )


def _numeric_field_value(
    inspected: InspectedHand,
    field: str,
) -> float | None:
    if field in (
        "pot",
        "max_pot",
    ):
        return (
            inspected.max_pot_bb
        )

    if field == "hero_profit":
        return (
            inspected.hero_profit_bb
        )

    if field == "hero_loss":
        if (
            inspected.hero_profit_bb
            is None
        ):
            return None

        return max(
            0.0,
            -inspected.hero_profit_bb,
        )

    raise ValueError(
        f"Unsupported numeric field: {field}"
    )


def _compare_numeric(
    actual: float,
    operator: str,
    expected: float,
) -> bool:
    if operator == ">":
        return actual > expected

    if operator == ">=":
        return actual >= expected

    if operator == "<":
        return actual < expected

    if operator == "<=":
        return actual <= expected

    if operator == "==":
        return actual == expected

    if operator == "!=":
        return actual != expected

    raise ValueError(
        "Unsupported numeric operator: "
        f"{operator}"
    )


def _parse_string_literal(
    raw: str,
) -> str:
    value = raw.strip()

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in (
            "'",
            '"',
        )
    ):
        return value[
            1:-1
        ]

    return value


def _parse_numeric_literal(
    raw: str,
) -> float:
    try:
        value = float(
            raw.strip()
        )

    except ValueError as exc:
        raise ValueError(
            "Expected numeric filter value: "
            f"{raw}"
        ) from exc

    if not math.isfinite(
        value
    ):
        raise ValueError(
            "Numeric filter value must be finite"
        )

    return value


def _seat_for_bot(
    hand: RecordedHand,
    bot_id: int,
) -> int | None:
    if (
        hand.seat0_bot_id
        == bot_id
    ):
        return 0

    if (
        hand.seat1_bot_id
        == bot_id
    ):
        return 1

    return None


def _street_name(
    street: Street,
) -> str:
    return (
        street.name.lower()
    )


def _position_name(
    dealer: int,
    seat: int,
) -> str:
    if dealer == seat:
        return "BTN/SB"

    return "BB"


def _actions_text(
    actions: tuple[
        Action,
        ...,
    ],
) -> str:
    if not actions:
        return "[]"

    return (
        "["
        + ", ".join(
            action.value
            for action
            in actions
        )
        + "]"
    )


def _format_optional_bb(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{value:.2f} BB"
    )


def _html_row(
    label: str,
    value: str,
) -> str:
    return (
        "<tr>"
        "<th>"
        f"{html.escape(label)}"
        "</th>"
        "<td>"
        f"{html.escape(value)}"
        "</td>"
        "</tr>"
    )