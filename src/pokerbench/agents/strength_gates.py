from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import StrEnum
from typing import Any, Iterable

from pokerbench.agents.equity import (
    EQUITY_RANGE_VERSION,
    EQUITY_VERSION,
)
from pokerbench.agents.profiles import (
    OpponentParameters,
    OpponentProfile,
)


STRENGTH_REFERENCE_VERSION = "policy-strength-reference-v1"
GATE_MAP_VERSION = "policy-strength-gates-v1"
PARAMETRIC_POLICY_VERSION = "parametric-policy-v1"
REFERENCE_OPPORTUNITIES_PER_SCENARIO = 5_000
PPM_SCALE = 1_000_000


class GateDirection(StrEnum):
    HIGH = "high"
    LOW = "low"


def _validate_ppm(
    value: int,
    *,
    field_name: str,
    upper_inclusive: int = PPM_SCALE,
) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")

    if not 0 <= value <= upper_inclusive:
        raise ValueError(
            f"{field_name} must be within [0, {upper_inclusive}]"
        )


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _decimal_probability(value: float | int) -> Decimal:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise TypeError("target_probability must be numeric")

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError("target_probability must be finite")

    if not 0.0 <= numeric <= 1.0:
        raise ValueError("target_probability must be within [0, 1]")

    return Decimal(str(value))


def _round_decimal_to_ppm(probability: Decimal) -> int:
    ppm = (
        probability * Decimal(PPM_SCALE)
    ).quantize(
        Decimal("1"),
        rounding=ROUND_HALF_UP,
    )

    value = int(ppm)
    return min(PPM_SCALE, max(0, value))


@dataclass(frozen=True)
class StrengthGate:
    direction: GateDirection
    threshold_ppm: int
    tie_probability_ppm: int

    def __post_init__(self) -> None:
        if not isinstance(self.direction, GateDirection):
            raise TypeError("direction must be a GateDirection")

        _validate_ppm(
            self.threshold_ppm,
            field_name="threshold_ppm",
        )
        _validate_ppm(
            self.tie_probability_ppm,
            field_name="tie_probability_ppm",
        )

    def passes(
        self,
        *,
        equity_ppm: int,
        decision_draw_ppm: int,
    ) -> bool:
        _validate_ppm(
            equity_ppm,
            field_name="equity_ppm",
        )

        if (
            not isinstance(decision_draw_ppm, int)
            or isinstance(decision_draw_ppm, bool)
        ):
            raise TypeError(
                "decision_draw_ppm must be an integer"
            )

        if not 0 <= decision_draw_ppm < PPM_SCALE:
            raise ValueError(
                "decision_draw_ppm must be within [0, 999999]"
            )

        if self.direction is GateDirection.HIGH:
            if equity_ppm > self.threshold_ppm:
                return True
            if equity_ppm < self.threshold_ppm:
                return False
        else:
            if equity_ppm < self.threshold_ppm:
                return True
            if equity_ppm > self.threshold_ppm:
                return False

        return (
            decision_draw_ppm
            < self.tie_probability_ppm
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "threshold_ppm": self.threshold_ppm,
            "tie_probability_ppm": (
                self.tie_probability_ppm
            ),
        }


def _validate_reference_strengths(
    values: tuple[int, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{field_name} must be a tuple")

    if len(values) != REFERENCE_OPPORTUNITIES_PER_SCENARIO:
        raise ValueError(
            f"{field_name} must contain exactly "
            f"{REFERENCE_OPPORTUNITIES_PER_SCENARIO} strengths"
        )

    for value in values:
        _validate_ppm(
            value,
            field_name=f"{field_name} strength",
        )


@dataclass(frozen=True)
class PolicyStrengthReference:
    open_btn: tuple[int, ...]
    three_bet_bb: tuple[int, ...]
    cbet_btn: tuple[int, ...]
    fold_to_cbet_bb: tuple[int, ...]
    postflop_aggression_btn: tuple[int, ...]
    version: str = STRENGTH_REFERENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != STRENGTH_REFERENCE_VERSION:
            raise ValueError(
                "Unsupported strength-reference version: "
                f"{self.version}"
            )

        for field_name in (
            "open_btn",
            "three_bet_bb",
            "cbet_btn",
            "fold_to_cbet_bb",
            "postflop_aggression_btn",
        ):
            _validate_reference_strengths(
                getattr(self, field_name),
                field_name=field_name,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "open_btn": list(self.open_btn),
            "three_bet_bb": list(self.three_bet_bb),
            "cbet_btn": list(self.cbet_btn),
            "fold_to_cbet_bb": list(
                self.fold_to_cbet_bb
            ),
            "postflop_aggression_btn": list(
                self.postflop_aggression_btn
            ),
        }

    @property
    def reference_strength_hash(self) -> str:
        return _sha256_payload(self.to_dict())


def build_strength_gate(
    strengths: Iterable[int],
    *,
    target_probability: float | int,
    direction: GateDirection,
) -> StrengthGate:
    if not isinstance(direction, GateDirection):
        raise TypeError("direction must be a GateDirection")

    values = tuple(strengths)

    if not values:
        raise ValueError("strengths must not be empty")

    for value in values:
        _validate_ppm(
            value,
            field_name="strength",
        )

    target = _decimal_probability(target_probability)

    if target == Decimal("0"):
        return StrengthGate(
            direction=direction,
            threshold_ppm=(
                PPM_SCALE
                if direction is GateDirection.HIGH
                else 0
            ),
            tie_probability_ppm=0,
        )

    if target == Decimal("1"):
        return StrengthGate(
            direction=direction,
            threshold_ppm=(
                0
                if direction is GateDirection.HIGH
                else PPM_SCALE
            ),
            tie_probability_ppm=PPM_SCALE,
        )

    counts = Counter(values)
    unique_strengths = sorted(
        counts,
        reverse=(direction is GateDirection.HIGH),
    )
    target_count = target * Decimal(len(values))
    strict_tail_count = 0

    for threshold in unique_strengths:
        equal_count = counts[threshold]
        lower = Decimal(strict_tail_count)
        upper = Decimal(strict_tail_count + equal_count)

        if lower <= target_count <= upper:
            tie_probability = (
                target_count - lower
            ) / Decimal(equal_count)

            return StrengthGate(
                direction=direction,
                threshold_ppm=threshold,
                tie_probability_ppm=(
                    _round_decimal_to_ppm(
                        tie_probability
                    )
                ),
            )

        strict_tail_count += equal_count

    raise RuntimeError(
        "Could not construct strength gate from reference distribution"
    )


def aggression_to_action_share(
    aggression: float | int,
) -> float:
    if (
        not isinstance(aggression, (int, float))
        or isinstance(aggression, bool)
    ):
        raise TypeError("aggression must be numeric")

    numeric = float(aggression)

    if not math.isfinite(numeric):
        raise ValueError("aggression must be finite")

    if numeric < 0.0:
        raise ValueError("aggression must be non-negative")

    value = Decimal(str(aggression))
    share = value / (Decimal("1") + value)
    return float(share)


def _required_frequency(
    value: float | None,
    *,
    field_name: str,
) -> float:
    if value is None:
        raise ValueError(
            f"{field_name} is required by parametric-policy-v1"
        )

    return value


def _high_gate_at_least_as_selective(
    inner: StrengthGate,
    outer: StrengthGate,
) -> bool:
    if (
        inner.direction is not GateDirection.HIGH
        or outer.direction is not GateDirection.HIGH
    ):
        raise ValueError(
            "Nested selectivity validation requires high-tail gates"
        )

    if inner.threshold_ppm > outer.threshold_ppm:
        return True

    if inner.threshold_ppm < outer.threshold_ppm:
        return False

    return (
        inner.tie_probability_ppm
        <= outer.tie_probability_ppm
    )


def _parameters_dict(
    parameters: OpponentParameters,
) -> dict[str, Any]:
    return {
        "vpip": parameters.vpip,
        "pfr": parameters.pfr,
        "three_bet_freq": parameters.three_bet_freq,
        "cbet_freq": parameters.cbet_freq,
        "aggression": parameters.aggression,
        "fold_to_cbet": parameters.fold_to_cbet,
        "schema_version": parameters.schema_version,
    }


@dataclass(frozen=True)
class PolicyCalibrationMap:
    parameters: OpponentParameters
    vpip_gate: StrengthGate
    pfr_gate: StrengthGate
    three_bet_gate: StrengthGate
    cbet_gate: StrengthGate
    aggression_gate: StrengthGate | None
    fold_to_cbet_gate: StrengthGate
    reference_strength_hash: str
    profile_name: str | None = None
    profile_version: str | None = None
    policy_version: str = PARAMETRIC_POLICY_VERSION
    strength_reference_version: str = STRENGTH_REFERENCE_VERSION
    gate_map_version: str = GATE_MAP_VERSION
    equity_version: str = EQUITY_VERSION
    equity_range_version: str = EQUITY_RANGE_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != PARAMETRIC_POLICY_VERSION:
            raise ValueError(
                "Unsupported parametric-policy version"
            )

        if (
            self.strength_reference_version
            != STRENGTH_REFERENCE_VERSION
        ):
            raise ValueError(
                "Unsupported strength-reference version"
            )

        if self.gate_map_version != GATE_MAP_VERSION:
            raise ValueError(
                "Unsupported gate-map version"
            )

        if self.equity_version != EQUITY_VERSION:
            raise ValueError(
                "Unsupported equity version"
            )

        if self.equity_range_version != EQUITY_RANGE_VERSION:
            raise ValueError(
                "Unsupported equity-range version"
            )

        if len(self.reference_strength_hash) != 64:
            raise ValueError(
                "reference_strength_hash must be a SHA-256 hex digest"
            )

        try:
            int(self.reference_strength_hash, 16)
        except ValueError as exc:
            raise ValueError(
                "reference_strength_hash must be a SHA-256 hex digest"
            ) from exc

        if (self.profile_name is None) != (
            self.profile_version is None
        ):
            raise ValueError(
                "profile_name and profile_version must be supplied together"
            )

        if not _high_gate_at_least_as_selective(
            self.pfr_gate,
            self.vpip_gate,
        ):
            raise ValueError(
                "PFR gate must be at least as selective as VPIP gate"
            )

    @property
    def generic_aggression_share_ppm(self) -> int:
        if self.parameters.aggression is None:
            return 0

        share = Decimal(str(self.parameters.aggression)) / (
            Decimal("1")
            + Decimal(str(self.parameters.aggression))
        )
        return _round_decimal_to_ppm(share)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "strength_reference_version": (
                self.strength_reference_version
            ),
            "gate_map_version": self.gate_map_version,
            "equity_version": self.equity_version,
            "equity_range_version": self.equity_range_version,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "parameters": _parameters_dict(self.parameters),
            "reference_strength_hash": (
                self.reference_strength_hash
            ),
            "vpip_gate": self.vpip_gate.to_dict(),
            "pfr_gate": self.pfr_gate.to_dict(),
            "three_bet_gate": self.three_bet_gate.to_dict(),
            "cbet_gate": self.cbet_gate.to_dict(),
            "aggression_gate": (
                None
                if self.aggression_gate is None
                else self.aggression_gate.to_dict()
            ),
            "generic_aggression_share_ppm": (
                self.generic_aggression_share_ppm
            ),
            "fold_to_cbet_gate": (
                self.fold_to_cbet_gate.to_dict()
            ),
        }

    @property
    def gate_map_hash(self) -> str:
        return _sha256_payload(self.to_dict())


def build_policy_calibration_map(
    parameters: OpponentParameters,
    reference: PolicyStrengthReference,
    *,
    profile_name: str | None = None,
    profile_version: str | None = None,
) -> PolicyCalibrationMap:
    three_bet_freq = _required_frequency(
        parameters.three_bet_freq,
        field_name="three_bet_freq",
    )
    cbet_freq = _required_frequency(
        parameters.cbet_freq,
        field_name="cbet_freq",
    )
    fold_to_cbet = _required_frequency(
        parameters.fold_to_cbet,
        field_name="fold_to_cbet",
    )

    vpip_gate = build_strength_gate(
        reference.open_btn,
        target_probability=parameters.vpip,
        direction=GateDirection.HIGH,
    )
    pfr_gate = build_strength_gate(
        reference.open_btn,
        target_probability=parameters.pfr,
        direction=GateDirection.HIGH,
    )

    if not _high_gate_at_least_as_selective(
        pfr_gate,
        vpip_gate,
    ):
        raise RuntimeError(
            "Constructed PFR gate is not nested inside VPIP gate"
        )

    aggression_gate: StrengthGate | None

    if parameters.aggression is None:
        aggression_gate = None
    else:
        aggression_gate = build_strength_gate(
            reference.postflop_aggression_btn,
            target_probability=(
                aggression_to_action_share(
                    parameters.aggression
                )
            ),
            direction=GateDirection.HIGH,
        )

    return PolicyCalibrationMap(
        parameters=parameters,
        vpip_gate=vpip_gate,
        pfr_gate=pfr_gate,
        three_bet_gate=build_strength_gate(
            reference.three_bet_bb,
            target_probability=three_bet_freq,
            direction=GateDirection.HIGH,
        ),
        cbet_gate=build_strength_gate(
            reference.cbet_btn,
            target_probability=cbet_freq,
            direction=GateDirection.HIGH,
        ),
        aggression_gate=aggression_gate,
        fold_to_cbet_gate=build_strength_gate(
            reference.fold_to_cbet_bb,
            target_probability=fold_to_cbet,
            direction=GateDirection.LOW,
        ),
        reference_strength_hash=(
            reference.reference_strength_hash
        ),
        profile_name=profile_name,
        profile_version=profile_version,
    )


def build_profile_calibration_map(
    profile: OpponentProfile,
    reference: PolicyStrengthReference,
) -> PolicyCalibrationMap:
    return build_policy_calibration_map(
        profile.parameters,
        reference,
        profile_name=profile.name.value,
        profile_version=profile.version,
    )
