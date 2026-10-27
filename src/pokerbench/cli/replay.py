"""Public deterministic match replay presentation helpers."""

from __future__ import annotations

from pathlib import Path

from pokerbench.cli.inspect import (
    InspectOptions,
    InspectedHand,
    inspect_match,
    render_inspection_html,
    render_inspection_text,
    write_inspection_html,
)
from pokerbench.replay import MatchRecord


def replay_match(
    match_record: MatchRecord,
    *,
    hero_bot_id: int = 0,
) -> tuple[InspectedHand, ...]:
    """Deterministically replay every persisted hand in one match."""
    return inspect_match(
        match_record,
        InspectOptions(
            hero_bot_id=hero_bot_id,
        ),
    )


def render_replay_text(
    match_record: MatchRecord,
    hands: tuple[InspectedHand, ...],
    *,
    hero_bot_id: int = 0,
) -> str:
    """Render the complete replay decision chain as text."""
    return render_inspection_text(
        match_record,
        hands,
        hero_bot_id=hero_bot_id,
    )


def render_replay_html(
    match_record: MatchRecord,
    hands: tuple[InspectedHand, ...],
    *,
    hero_bot_id: int = 0,
) -> str:
    """Render the minimal self-contained HTML replay viewer."""
    return render_inspection_html(
        match_record,
        hands,
        hero_bot_id=hero_bot_id,
    )


def write_replay_html(
    path: str | Path,
    content: str,
) -> Path:
    """Write replay HTML using the established viewer persistence helper."""
    return write_inspection_html(
        path,
        content,
    )
