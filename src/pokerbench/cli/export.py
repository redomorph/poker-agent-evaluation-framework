"""Read-only deterministic benchmark export bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Mapping, Sequence

from pokerbench.cli.inspect import InspectOptions, inspect_match
from pokerbench.engine.deal_schedule import DealSchedule
from pokerbench.persistence import (
    MatchNotFoundError,
    MatchPersistenceMetadata,
    SQLITE_SCHEMA_MIGRATION_ID,
    SQLITE_SCHEMA_VERSION,
    SQLiteBenchmarkStore,
)
from pokerbench.replay import MatchRecord, RecordedHand


PUBLIC_EXPORT_CLI_VERSION = "public-export-cli-v1"
EXPORT_BUNDLE_VERSION = "benchmark-export-bundle-v1"

_EXPORT_FILES = (
    "records.jsonl",
    "matches.csv",
    "hands.csv",
    "decisions.csv",
    "timeouts.csv",
)

_MATCH_FIELDS = (
    "match_id",
    "match_record_schema_version",
    "master_seed",
    "experiment_id",
    "experiment_config_hash",
    "game_config_hash",
    "record_hash",
    "abstraction_version",
    "starting_stack_bb",
    "small_blind_bb",
    "big_blind_bb",
    "rake",
    "hands_played",
    "bot_version",
    "opponent_hash",
    "ev_bb_100",
    "ci_low",
    "ci_high",
    "engine_version",
    "python_version",
    "rng_version",
    "git_commit",
    "population_id",
    "population_seed",
    "opponent_parameters_json",
    "bot_configuration_json",
    "statistical_method",
    "sample_size",
    "precision_target_json",
    "experiment_config_json",
    "poetry_lock_hash",
    "hardware_fingerprint",
    "os_identifier",
)

_HAND_FIELDS = (
    "match_id",
    "hand_ordinal",
    "config_id",
    "master_seed",
    "hand_index",
    "dealer",
    "seat0_bot_id",
    "seat1_bot_id",
    "seat0_hole_cards_json",
    "seat1_hole_cards_json",
    "full_board_json",
    "actions_json",
    "termination",
    "terminal_profit_seat0_bb",
    "terminal_profit_seat1_bb",
    "requested_action",
    "legal_actions_at_termination_json",
    "acting_player_at_termination",
    "failure_message",
    "timeout_count",
    "max_pot_bb",
    "hand_record_hash",
)

_DECISION_FIELDS = (
    "match_id",
    "hand_ordinal",
    "hand_index",
    "action_index",
    "actor_seat",
    "actor_bot_id",
    "street",
    "position",
    "pot_bb",
    "stack0_bb",
    "stack1_bb",
    "current_bet0_bb",
    "current_bet1_bb",
    "full_board_json",
    "visible_board_json",
    "actor_hole_cards_json",
    "opponent_hole_cards_json",
    "action_history_json",
    "legal_actions_json",
    "chosen_action",
)

_TIMEOUT_FIELDS = (
    "match_id",
    "hand_ordinal",
    "hand_index",
    "timeout_ordinal",
    "player",
    "bot_id",
    "action_index",
    "decision_index",
    "elapsed_ns",
    "budget_ms",
    "timeout_policy",
    "timeout_policy_version",
    "resolution",
    "applied_action",
)


class PublicExportError(ValueError):
    """Raised when a deterministic public export cannot be produced."""


@dataclass(frozen=True)
class PublicExportResult:
    """Summary of one committed export bundle."""

    output: Path
    selected_match_ids: tuple[str, ...]
    match_count: int
    hand_count: int
    decision_count: int
    timeout_count: int
    bundle_hash: str
    public_export_cli_version: str = PUBLIC_EXPORT_CLI_VERSION
    export_bundle_version: str = EXPORT_BUNDLE_VERSION


def execute_public_export(
    *,
    database: Path,
    output: Path,
    match_ids: Sequence[str] | None = None,
) -> PublicExportResult:
    """Export selected persisted matches into one atomic deterministic bundle."""
    destination = Path(output)
    if destination.exists():
        raise PublicExportError(
            f"Export destination already exists: {destination}"
        )

    requested = tuple(match_ids or ())
    if len(set(requested)) != len(requested):
        raise PublicExportError("Duplicate --match-id values are not allowed")
    if any(not match_id.strip() for match_id in requested):
        raise PublicExportError("match IDs must not be empty")

    store = SQLiteBenchmarkStore(database)
    available = store.list_match_ids()

    if requested:
        selected = tuple(sorted(requested))
        available_set = set(available)
        missing = tuple(match_id for match_id in selected if match_id not in available_set)
        if missing:
            raise MatchNotFoundError(
                "Match not found: " + ", ".join(missing)
            )
    else:
        selected = available

    if not selected:
        raise PublicExportError("No persisted matches are available for export")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )

    try:
        result = _write_bundle(
            store=store,
            selected_match_ids=selected,
            directory=temporary,
        )
        if destination.exists():
            raise PublicExportError(
                f"Export destination already exists: {destination}"
            )
        os.rename(temporary, destination)
        return PublicExportResult(
            output=destination,
            selected_match_ids=result.selected_match_ids,
            match_count=result.match_count,
            hand_count=result.hand_count,
            decision_count=result.decision_count,
            timeout_count=result.timeout_count,
            bundle_hash=result.bundle_hash,
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def render_public_export_text(result: PublicExportResult) -> str:
    """Render a compact human-readable export summary."""
    return "\n".join(
        [
            "PokerBench export complete",
            f"  output: {result.output}",
            f"  matches: {result.match_count}",
            f"  hands: {result.hand_count}",
            f"  decisions: {result.decision_count}",
            f"  timeouts: {result.timeout_count}",
            f"  bundle hash: {result.bundle_hash}",
        ]
    )


def _write_bundle(
    *,
    store: SQLiteBenchmarkStore,
    selected_match_ids: tuple[str, ...],
    directory: Path,
) -> PublicExportResult:
    records_path = directory / "records.jsonl"
    matches_path = directory / "matches.csv"
    hands_path = directory / "hands.csv"
    decisions_path = directory / "decisions.csv"
    timeouts_path = directory / "timeouts.csv"

    match_count = 0
    hand_count = 0
    decision_count = 0
    timeout_count = 0

    with (
        records_path.open("w", encoding="utf-8", newline="\n") as records_handle,
        _csv_writer(matches_path, _MATCH_FIELDS) as matches_writer,
        _csv_writer(hands_path, _HAND_FIELDS) as hands_writer,
        _csv_writer(decisions_path, _DECISION_FIELDS) as decisions_writer,
        _csv_writer(timeouts_path, _TIMEOUT_FIELDS) as timeouts_writer,
    ):
        for match_id in selected_match_ids:
            record = store.load_match(match_id)
            metadata = store.load_match_metadata(match_id)
            _validate_record_metadata_identity(record, metadata)

            records_handle.write(_canonical_json(record.to_dict()))
            records_handle.write("\n")
            matches_writer.writerow(_match_row(record, metadata))
            match_count += 1

            inspected_hands = inspect_match(
                record,
                InspectOptions(hero_bot_id=0),
            )
            if len(inspected_hands) != len(record.hands):
                raise PublicExportError(
                    f"Replay inspection hand count mismatch for {match_id}"
                )

            for hand_ordinal, (hand, inspected) in enumerate(
                zip(record.hands, inspected_hands, strict=True)
            ):
                hands_writer.writerow(
                    _hand_row(
                        record=record,
                        hand=hand,
                        hand_ordinal=hand_ordinal,
                        max_pot_bb=inspected.max_pot_bb,
                    )
                )
                hand_count += 1

                for decision in inspected.decisions:
                    decisions_writer.writerow(
                        {
                            "match_id": record.match_id,
                            "hand_ordinal": hand_ordinal,
                            "hand_index": hand.hand_index,
                            "action_index": decision.action_index,
                            "actor_seat": decision.actor_seat,
                            "actor_bot_id": decision.actor_bot_id,
                            "street": decision.street,
                            "position": decision.position,
                            "pot_bb": decision.pot_bb,
                            "stack0_bb": decision.stacks_bb[0],
                            "stack1_bb": decision.stacks_bb[1],
                            "current_bet0_bb": decision.current_bets_bb[0],
                            "current_bet1_bb": decision.current_bets_bb[1],
                            "full_board_json": _canonical_json(list(decision.full_board)),
                            "visible_board_json": _canonical_json(list(decision.visible_board)),
                            "actor_hole_cards_json": _canonical_json(list(decision.actor_hole_cards)),
                            "opponent_hole_cards_json": _canonical_json(list(decision.opponent_hole_cards)),
                            "action_history_json": _canonical_json(
                                [action.value for action in decision.action_history]
                            ),
                            "legal_actions_json": _canonical_json(
                                [action.value for action in decision.legal_actions]
                            ),
                            "chosen_action": decision.chosen_action.value,
                        }
                    )
                    decision_count += 1

                for timeout_ordinal, event in enumerate(hand.timeout_events):
                    timeouts_writer.writerow(
                        {
                            "match_id": record.match_id,
                            "hand_ordinal": hand_ordinal,
                            "hand_index": hand.hand_index,
                            "timeout_ordinal": timeout_ordinal,
                            "player": event.player,
                            "bot_id": event.bot_id,
                            "action_index": event.action_index,
                            "decision_index": event.decision_index,
                            "elapsed_ns": event.elapsed_ns,
                            "budget_ms": event.budget_ms,
                            "timeout_policy": event.timeout_policy,
                            "timeout_policy_version": event.timeout_policy_version,
                            "resolution": event.resolution,
                            "applied_action": (
                                event.applied_action.value
                                if event.applied_action is not None
                                else None
                            ),
                        }
                    )
                    timeout_count += 1

    file_hashes = {
        name: _sha256_file(directory / name)
        for name in _EXPORT_FILES
    }
    manifest_identity = {
        "public_export_cli_version": PUBLIC_EXPORT_CLI_VERSION,
        "export_bundle_version": EXPORT_BUNDLE_VERSION,
        "sqlite_schema_version": SQLITE_SCHEMA_VERSION,
        "sqlite_schema_migration_id": SQLITE_SCHEMA_MIGRATION_ID,
        "selected_match_ids": list(selected_match_ids),
        "match_count": match_count,
        "hand_count": hand_count,
        "decision_count": decision_count,
        "timeout_count": timeout_count,
        "files": file_hashes,
    }
    bundle_hash = _sha256_text(_canonical_json(manifest_identity))
    manifest = dict(manifest_identity)
    manifest["bundle_hash"] = bundle_hash
    (directory / "manifest.json").write_text(
        _canonical_json(manifest) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return PublicExportResult(
        output=directory,
        selected_match_ids=selected_match_ids,
        match_count=match_count,
        hand_count=hand_count,
        decision_count=decision_count,
        timeout_count=timeout_count,
        bundle_hash=bundle_hash,
    )


class _CSVWriterContext:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self._path = path
        self._fieldnames = tuple(fieldnames)
        self._handle = None
        self._writer = None

    def __enter__(self) -> csv.DictWriter:
        self._handle = self._path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=self._fieldnames,
            lineterminator="\n",
            extrasaction="raise",
        )
        self._writer.writeheader()
        return self._writer

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type
        del exc
        del tb
        if self._handle is not None:
            self._handle.close()


def _csv_writer(path: Path, fieldnames: Sequence[str]) -> _CSVWriterContext:
    return _CSVWriterContext(path, fieldnames)


def _match_row(
    record: MatchRecord,
    metadata: MatchPersistenceMetadata,
) -> dict[str, object]:
    config = record.game_config
    return {
        "match_id": record.match_id,
        "match_record_schema_version": record.schema_version,
        "master_seed": record.master_seed,
        "experiment_id": record.experiment_id,
        "experiment_config_hash": record.experiment_config_hash,
        "game_config_hash": record.game_config_hash,
        "record_hash": record.record_hash,
        "abstraction_version": config.abstraction_version,
        "starting_stack_bb": config.starting_stack_bb,
        "small_blind_bb": config.small_blind_bb,
        "big_blind_bb": config.big_blind_bb,
        "rake": config.rake,
        "hands_played": len(record.hands),
        "bot_version": metadata.bot_version,
        "opponent_hash": metadata.opponent_hash,
        "ev_bb_100": metadata.ev_bb_100,
        "ci_low": metadata.ci_low,
        "ci_high": metadata.ci_high,
        "engine_version": metadata.engine_version,
        "python_version": metadata.python_version,
        "rng_version": metadata.rng_version,
        "git_commit": metadata.git_commit,
        "population_id": metadata.population_id,
        "population_seed": metadata.population_seed,
        "opponent_parameters_json": _optional_canonical_mapping(metadata.opponent_parameters),
        "bot_configuration_json": _optional_canonical_mapping(metadata.bot_configuration),
        "statistical_method": metadata.statistical_method,
        "sample_size": metadata.sample_size,
        "precision_target_json": _optional_canonical_mapping(metadata.precision_target),
        "experiment_config_json": _optional_canonical_mapping(metadata.experiment_config),
        "poetry_lock_hash": metadata.poetry_lock_hash,
        "hardware_fingerprint": metadata.hardware_fingerprint,
        "os_identifier": metadata.os_identifier,
    }


def _hand_row(
    *,
    record: MatchRecord,
    hand: RecordedHand,
    hand_ordinal: int,
    max_pot_bb: float,
) -> dict[str, object]:
    seat0_cards, seat1_cards, board = DealSchedule.hand_for_index(
        hand.master_seed,
        hand.hand_index,
    )
    profit = hand.terminal_profit_bb
    return {
        "match_id": record.match_id,
        "hand_ordinal": hand_ordinal,
        "config_id": hand.config_id,
        "master_seed": hand.master_seed,
        "hand_index": hand.hand_index,
        "dealer": hand.dealer,
        "seat0_bot_id": hand.seat0_bot_id,
        "seat1_bot_id": hand.seat1_bot_id,
        "seat0_hole_cards_json": _canonical_json(list(seat0_cards)),
        "seat1_hole_cards_json": _canonical_json(list(seat1_cards)),
        "full_board_json": _canonical_json(list(board)),
        "actions_json": _canonical_json([action.value for action in hand.actions]),
        "termination": hand.termination.value,
        "terminal_profit_seat0_bb": profit[0] if profit is not None else None,
        "terminal_profit_seat1_bb": profit[1] if profit is not None else None,
        "requested_action": hand.requested_action.value if hand.requested_action is not None else None,
        "legal_actions_at_termination_json": _canonical_json(
            [action.value for action in hand.legal_actions_at_termination]
        ),
        "acting_player_at_termination": hand.acting_player_at_termination,
        "failure_message": hand.failure_message,
        "timeout_count": len(hand.timeout_events),
        "max_pot_bb": max_pot_bb,
        "hand_record_hash": _sha256_text(_canonical_json(hand.to_dict())),
    }


def _validate_record_metadata_identity(
    record: MatchRecord,
    metadata: MatchPersistenceMetadata,
) -> None:
    if metadata.experiment_config is None:
        return
    calculated = _sha256_text(_canonical_json(metadata.experiment_config))
    if record.experiment_config_hash != calculated:
        raise PublicExportError(
            f"Experiment configuration identity mismatch for {record.match_id}"
        )


def _optional_canonical_mapping(value: Mapping[str, object] | None) -> str | None:
    if value is None:
        return None
    return _canonical_json(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
