from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pokerbench.agents.profiles import (
    CALLING_STATION_PROFILE,
    STATIC_PROFILES,
    TAG_PROFILE,
    OpponentParameters,
)
from pokerbench.agents.strength_gates import (
    GATE_MAP_VERSION,
    PPM_SCALE,
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    STRENGTH_REFERENCE_VERSION,
    GateDirection,
    PolicyCalibrationMap,
    PolicyStrengthReference,
    StrengthGate,
    aggression_to_action_share,
    build_policy_calibration_map,
    build_profile_calibration_map,
    build_strength_gate,
)


def _values(
    *groups: tuple[int, int],
) -> tuple[int, ...]:
    values: list[int] = []

    for strength, count in groups:
        values.extend([strength] * count)

    assert len(values) == REFERENCE_OPPORTUNITIES_PER_SCENARIO
    return tuple(values)


def _reference() -> PolicyStrengthReference:
    opening = _values(
        (900_000, 1_000),
        (500_000, 2_000),
        (100_000, 2_000),
    )
    three_bet = _values(
        (800_000, 500),
        (400_000, 2_000),
        (200_000, 2_500),
    )
    cbet = _values(
        (850_000, 1_500),
        (550_000, 2_000),
        (250_000, 1_500),
    )
    fold_to_cbet = _values(
        (800_000, 1_000),
        (500_000, 2_000),
        (150_000, 2_000),
    )
    aggression = _values(
        (600_000, 5_000),
    )

    return PolicyStrengthReference(
        open_btn=opening,
        three_bet_bb=three_bet,
        cbet_btn=cbet,
        fold_to_cbet_bb=fold_to_cbet,
        postflop_aggression_btn=aggression,
    )


def test_versions_and_reference_size_are_frozen() -> None:
    assert STRENGTH_REFERENCE_VERSION == "policy-strength-reference-v1"
    assert GATE_MAP_VERSION == "policy-strength-gates-v1"
    assert REFERENCE_OPPORTUNITIES_PER_SCENARIO == 5_000
    assert PPM_SCALE == 1_000_000


def test_reference_is_frozen_validated_and_hash_stable() -> None:
    first = _reference()
    second = _reference()

    assert first.reference_strength_hash == second.reference_strength_hash
    assert len(first.reference_strength_hash) == 64

    with pytest.raises(FrozenInstanceError):
        first.open_btn = ()  # type: ignore[misc]

    with pytest.raises(ValueError, match="exactly 5000"):
        PolicyStrengthReference(
            open_btn=(500_000,),
            three_bet_bb=first.three_bet_bb,
            cbet_btn=first.cbet_btn,
            fold_to_cbet_bb=first.fold_to_cbet_bb,
            postflop_aggression_btn=first.postflop_aggression_btn,
        )

    invalid = list(first.open_btn)
    invalid[0] = 1_000_001

    with pytest.raises(ValueError, match="within"):
        PolicyStrengthReference(
            open_btn=tuple(invalid),
            three_bet_bb=first.three_bet_bb,
            cbet_btn=first.cbet_btn,
            fold_to_cbet_bb=first.fold_to_cbet_bb,
            postflop_aggression_btn=first.postflop_aggression_btn,
        )


def test_high_tail_gate_uses_exact_tie_probability() -> None:
    strengths = _reference().open_btn

    gate = build_strength_gate(
        strengths,
        target_probability=0.30,
        direction=GateDirection.HIGH,
    )

    assert gate == StrengthGate(
        direction=GateDirection.HIGH,
        threshold_ppm=500_000,
        tie_probability_ppm=250_000,
    )

    assert gate.passes(
        equity_ppm=900_000,
        decision_draw_ppm=999_999,
    )
    assert gate.passes(
        equity_ppm=500_000,
        decision_draw_ppm=249_999,
    )
    assert not gate.passes(
        equity_ppm=500_000,
        decision_draw_ppm=250_000,
    )
    assert not gate.passes(
        equity_ppm=100_000,
        decision_draw_ppm=0,
    )


def test_low_tail_gate_uses_exact_tie_probability() -> None:
    strengths = _reference().fold_to_cbet_bb

    gate = build_strength_gate(
        strengths,
        target_probability=0.50,
        direction=GateDirection.LOW,
    )

    assert gate == StrengthGate(
        direction=GateDirection.LOW,
        threshold_ppm=500_000,
        tie_probability_ppm=250_000,
    )

    assert gate.passes(
        equity_ppm=150_000,
        decision_draw_ppm=999_999,
    )
    assert gate.passes(
        equity_ppm=500_000,
        decision_draw_ppm=249_999,
    )
    assert not gate.passes(
        equity_ppm=500_000,
        decision_draw_ppm=250_000,
    )
    assert not gate.passes(
        equity_ppm=800_000,
        decision_draw_ppm=0,
    )


def test_boundary_targets_use_frozen_representations() -> None:
    strengths = _reference().open_btn

    assert build_strength_gate(
        strengths,
        target_probability=0.0,
        direction=GateDirection.HIGH,
    ) == StrengthGate(
        GateDirection.HIGH,
        1_000_000,
        0,
    )

    assert build_strength_gate(
        strengths,
        target_probability=1.0,
        direction=GateDirection.HIGH,
    ) == StrengthGate(
        GateDirection.HIGH,
        0,
        1_000_000,
    )

    assert build_strength_gate(
        strengths,
        target_probability=0.0,
        direction=GateDirection.LOW,
    ) == StrengthGate(
        GateDirection.LOW,
        0,
        0,
    )

    assert build_strength_gate(
        strengths,
        target_probability=1.0,
        direction=GateDirection.LOW,
    ) == StrengthGate(
        GateDirection.LOW,
        1_000_000,
        1_000_000,
    )


def test_exact_interior_boundary_has_unique_tail_first_representation() -> None:
    strengths = _values(
        (900_000, 1_000),
        (700_000, 1_500),
        (300_000, 2_500),
    )

    high = build_strength_gate(
        strengths,
        target_probability=0.50,
        direction=GateDirection.HIGH,
    )
    low = build_strength_gate(
        strengths,
        target_probability=0.50,
        direction=GateDirection.LOW,
    )

    assert high == StrengthGate(
        GateDirection.HIGH,
        700_000,
        1_000_000,
    )
    assert low == StrengthGate(
        GateDirection.LOW,
        300_000,
        1_000_000,
    )


def test_gate_rejects_invalid_inputs() -> None:
    strengths = _reference().open_btn

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_strength_gate(
            strengths,
            target_probability=1.01,
            direction=GateDirection.HIGH,
        )

    with pytest.raises(ValueError, match="decision_draw_ppm"):
        StrengthGate(
            GateDirection.HIGH,
            500_000,
            500_000,
        ).passes(
            equity_ppm=500_000,
            decision_draw_ppm=1_000_000,
        )


def test_aggression_ratio_is_converted_to_aggressive_share() -> None:
    assert aggression_to_action_share(0.0) == 0.0
    assert aggression_to_action_share(1.0) == 0.5
    assert aggression_to_action_share(1.5) == 0.6

    with pytest.raises(ValueError, match="non-negative"):
        aggression_to_action_share(-0.1)



def test_aggression_gate_uses_ratio_to_share_target() -> None:
    gate_map = build_profile_calibration_map(
        TAG_PROFILE,
        _reference(),
    )

    assert gate_map.generic_aggression_share_ppm == 444_444
    assert gate_map.aggression_gate == StrengthGate(
        GateDirection.HIGH,
        600_000,
        444_444,
    )


def test_all_static_profiles_build_from_one_frozen_reference() -> None:
    reference = _reference()

    maps = tuple(
        build_profile_calibration_map(
            profile,
            reference,
        )
        for profile in STATIC_PROFILES.values()
    )

    assert len(maps) == 5
    assert len({item.gate_map_hash for item in maps}) == 5


def test_profile_map_builds_all_gates_and_disables_calling_station_aggression() -> None:
    reference = _reference()

    tag = build_profile_calibration_map(
        TAG_PROFILE,
        reference,
    )
    station = build_profile_calibration_map(
        CALLING_STATION_PROFILE,
        reference,
    )

    assert isinstance(tag, PolicyCalibrationMap)
    assert tag.profile_name == "tag"
    assert tag.profile_version == "static-profiles-v2"
    assert tag.reference_strength_hash == reference.reference_strength_hash
    assert tag.aggression_gate is not None

    assert station.profile_name == "calling_station"
    assert station.aggression_gate is None
    assert station.generic_aggression_share_ppm == 0
    assert station.cbet_gate.direction is GateDirection.HIGH


def test_pfr_gate_is_nested_inside_vpip_gate() -> None:
    gate_map = build_profile_calibration_map(
        TAG_PROFILE,
        _reference(),
    )

    for equity_ppm in (
        0,
        100_000,
        500_000,
        900_000,
        1_000_000,
    ):
        for draw in (
            0,
            249_999,
            500_000,
            999_999,
        ):
            if gate_map.pfr_gate.passes(
                equity_ppm=equity_ppm,
                decision_draw_ppm=draw,
            ):
                assert gate_map.vpip_gate.passes(
                    equity_ppm=equity_ppm,
                    decision_draw_ppm=draw,
                )


def test_generic_map_requires_all_non_dont_care_v1_targets() -> None:
    reference = _reference()
    incomplete = OpponentParameters(
        vpip=0.20,
        pfr=0.10,
        three_bet_freq=None,
        cbet_freq=0.50,
        aggression=0.50,
        fold_to_cbet=0.40,
    )

    with pytest.raises(ValueError, match="three_bet_freq"):
        build_policy_calibration_map(
            incomplete,
            reference,
        )


def test_gate_map_hash_is_canonical_and_policy_sensitive() -> None:
    reference = _reference()
    first = build_profile_calibration_map(
        TAG_PROFILE,
        reference,
    )
    second = build_profile_calibration_map(
        TAG_PROFILE,
        reference,
    )

    assert first.gate_map_hash == second.gate_map_hash
    assert len(first.gate_map_hash) == 64
    assert first.to_dict() == second.to_dict()

    changed = build_policy_calibration_map(
        OpponentParameters(
            vpip=0.23,
            pfr=0.18,
            three_bet_freq=0.07,
            cbet_freq=0.65,
            aggression=0.80,
            fold_to_cbet=0.45,
        ),
        reference,
    )

    assert changed.gate_map_hash != first.gate_map_hash
