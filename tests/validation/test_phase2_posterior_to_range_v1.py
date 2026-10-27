from __future__ import annotations

import json
from pathlib import Path

import pytest

from pokerbench.agents.opponent_model import (
    BayesianMetricPosterior,
    BayesianOpponentModel,
    BetaPrior,
    OpponentModelMetric,
    OpponentModelPrior,
)
from pokerbench.agents.opponent_range import (
    POSTERIOR_TO_RANGE_PROTOCOL_VERSION,
    PREFLOP_RANGE_STRENGTH_REFERENCE_VERSION,
    PreflopRangeSignal,
    canonical_class_combinations,
    classify_preflop_range_signal,
    construct_opponent_range,
    effective_preflop_widths_ppm,
    expand_reference_combinations,
    load_preflop_strength_reference,
    load_posterior_to_range_protocol,
)
from pokerbench.engine.config import Action
from pokerbench.engine.public_action import PublicActionEvent
from pokerbench.engine.state import Street

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/experiments/phase2-posterior-to-range-v1.json"
REFERENCE_PATH = ROOT / "configs/experiments/preflop-range-strength-reference-v1.json"


def _prior() -> OpponentModelPrior:
    one = BetaPrior(1, 1)
    return OpponentModelPrior(
        vpip=one,
        pfr=one,
        three_bet=one,
        cbet=one,
        aggressive_share=one,
        fold_to_cbet=one,
    )


def _posterior(
    metric: OpponentModelMetric,
    prior: BetaPrior,
    *,
    events: int,
    opportunities: int,
) -> BayesianMetricPosterior:
    return BayesianMetricPosterior(
        metric=metric,
        prior=prior,
        events=events,
        opportunities=opportunities,
    )


def _model(
    *,
    vpip: tuple[int, int] = (20, 100),
    pfr: tuple[int, int] = (30, 100),
    three_bet: tuple[int, int] = (40, 100),
    cbet: tuple[int, int] = (50, 100),
    aggressive: tuple[int, int] = (60, 100),
    fold_to_cbet: tuple[int, int] = (70, 100),
) -> BayesianOpponentModel:
    prior = _prior()
    return BayesianOpponentModel(
        prior=prior,
        hands_observed=vpip[1],
        vpip=_posterior(OpponentModelMetric.VPIP, prior.vpip, events=vpip[0], opportunities=vpip[1]),
        pfr=_posterior(OpponentModelMetric.PFR, prior.pfr, events=pfr[0], opportunities=pfr[1]),
        three_bet=_posterior(OpponentModelMetric.THREE_BET, prior.three_bet, events=three_bet[0], opportunities=three_bet[1]),
        cbet=_posterior(OpponentModelMetric.CBET, prior.cbet, events=cbet[0], opportunities=cbet[1]),
        aggressive_share=_posterior(OpponentModelMetric.AGGRESSIVE_SHARE, prior.aggressive_share, events=aggressive[0], opportunities=aggressive[1]),
        fold_to_cbet=_posterior(OpponentModelMetric.FOLD_TO_CBET, prior.fold_to_cbet, events=fold_to_cbet[0], opportunities=fold_to_cbet[1]),
    )


def _event(actor: int, action: Action, wager_increase: bool) -> PublicActionEvent:
    return PublicActionEvent(
        actor=actor,
        street=Street.PREFLOP,
        action=action,
        wager_increase=wager_increase,
    )


def test_protocol_and_reference_files_load_as_frozen_contracts() -> None:
    protocol = load_posterior_to_range_protocol(PROTOCOL_PATH, reference_path=REFERENCE_PATH)
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    assert protocol.protocol_version == POSTERIOR_TO_RANGE_PROTOCOL_VERSION
    assert reference.reference_version == PREFLOP_RANGE_STRENGTH_REFERENCE_VERSION
    assert reference.payload_hash == "cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e"
    assert len(reference.class_rows) == 169


def test_reference_expands_to_all_1326_unique_physical_combinations() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    combos = expand_reference_combinations(reference)
    assert len(combos) == 1326
    assert len(set(combos)) == 1326
    assert len(canonical_class_combinations("AA")) == 6
    assert len(canonical_class_combinations("AKs")) == 4
    assert len(canonical_class_combinations("AKo")) == 12


def test_reference_has_frozen_equity_anchors_and_strength_order() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    scores = {row.class_id: row.equity_ppm for row in reference.class_rows}
    assert scores["AA"] == 849750
    assert scores["AKs"] == 667750
    assert scores["AKo"] == 656700
    assert scores["72o"] == 345550
    assert scores["32o"] == 319950
    assert reference.strength_order[0] == "AA"
    assert reference.strength_order[-1] == "32o"


def test_effective_widths_use_posterior_means_and_enforce_nesting() -> None:
    widths = effective_preflop_widths_ppm(_model())
    # Raw posterior means are 21/102, 31/102, 41/102.
    # PFR and 3Bet are projected downward only for range consistency.
    assert widths.raw_vpip_ppm == 205882
    assert widths.raw_pfr_ppm == 303922
    assert widths.raw_three_bet_ppm == 401961
    assert widths.effective_vpip_ppm == 205882
    assert widths.effective_pfr_ppm == 205882
    assert widths.effective_three_bet_ppm == 205882


def test_preflop_public_line_classification_is_frozen() -> None:
    assert classify_preflop_range_signal((), opponent_id=1) is PreflopRangeSignal.UNINFORMATIVE
    assert classify_preflop_range_signal((_event(1, Action.CALL, False),), opponent_id=1) is PreflopRangeSignal.PASSIVE_ENTRY
    assert classify_preflop_range_signal((_event(1, Action.BET_100, True),), opponent_id=1) is PreflopRangeSignal.FIRST_RAISE
    assert classify_preflop_range_signal((_event(0, Action.BET_100, True), _event(1, Action.BET_200, True)), opponent_id=1) is PreflopRangeSignal.THREE_BET
    assert classify_preflop_range_signal((_event(1, Action.BET_100, True), _event(0, Action.BET_200, True), _event(1, Action.ALL_IN, True)), opponent_id=1) is PreflopRangeSignal.FIRST_RAISE


def test_quantile_selection_uses_ceil_over_full_1326_combinations() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    model = _model(vpip=(20, 100), pfr=(10, 100), three_bet=(5, 100))
    result = construct_opponent_range(model, PreflopRangeSignal.PASSIVE_ENTRY, reference)
    assert result.width_ppm == 205882
    assert result.selected_combo_count == 273  # ceil(205882 * 1326 / 1e6)
    assert len(result.selected_combinations) == 273


def test_blockers_filter_after_quantile_selection_without_backfill() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    result = construct_opponent_range(_model(), PreflopRangeSignal.UNINFORMATIVE, reference)
    legal = result.legal_combinations((0, 13))
    assert result.selected_combo_count == 1326
    assert len(legal) == 1225
    assert all(0 not in combo and 13 not in combo for combo in legal)


def test_postflop_behavior_posteriors_do_not_change_v1_card_range() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    a = _model(cbet=(1, 100), aggressive=(1, 100), fold_to_cbet=(1, 100))
    b = _model(cbet=(99, 100), aggressive=(99, 100), fold_to_cbet=(99, 100))
    ra = construct_opponent_range(a, PreflopRangeSignal.FIRST_RAISE, reference)
    rb = construct_opponent_range(b, PreflopRangeSignal.FIRST_RAISE, reference)
    assert ra.selected_combinations == rb.selected_combinations
    assert ra.width_ppm == rb.width_ppm
    assert ra.model_state_hash != rb.model_state_hash


def test_range_snapshot_hash_is_deterministic_and_model_bound() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    model = _model(vpip=(25, 100), pfr=(18, 100), three_bet=(7, 100))
    first = construct_opponent_range(model, PreflopRangeSignal.FIRST_RAISE, reference)
    second = construct_opponent_range(model, PreflopRangeSignal.FIRST_RAISE, reference)
    assert first == second
    assert first.model_state_hash == model.state_hash
    assert len(first.range_hash) == 64


def test_non_preflop_public_events_are_ignored_by_signal_classifier() -> None:
    flop = PublicActionEvent(actor=1, street=Street.FLOP, action=Action.BET_100, wager_increase=True)
    assert classify_preflop_range_signal((flop,), opponent_id=1) is PreflopRangeSignal.UNINFORMATIVE


def test_protocol_rejects_semantic_tampering_even_with_recomputed_outer_hash(tmp_path: Path) -> None:
    document = json.loads(PROTOCOL_PATH.read_bytes())
    document["payload"]["postflop"]["action_conditioned_narrowing"] = True
    import hashlib
    raw = json.dumps(document["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    document["payload_hash"] = hashlib.sha256(raw).hexdigest()
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n", encoding="ascii", newline="\n")
    with pytest.raises(ValueError):
        load_posterior_to_range_protocol(path, reference_path=REFERENCE_PATH)


def test_empty_blocked_range_is_an_error_not_silent_backfill() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    model = _model(vpip=(0, 2000), pfr=(0, 100), three_bet=(0, 100))
    result = construct_opponent_range(
        model,
        PreflopRangeSignal.PASSIVE_ENTRY,
        reference,
    )
    assert result.selected_combo_count == 1
    only_combo = result.selected_combinations[0]
    with pytest.raises(ValueError, match="forbids range backfill"):
        result.legal_combinations(only_combo)
