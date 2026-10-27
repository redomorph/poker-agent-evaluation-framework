"""Synchronous persisted execution and reporting for Phase-0 Gate 6."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pokerbench.agents.calibration_environment import (
    CALIBRATION_ENVIRONMENT_VERSION,
    FIT_BASE_SEED,
    VALIDATION_BASE_SEED,
    CalibrationPhase,
    CalibrationScenario,
)
from pokerbench.agents.profiles import STATIC_PROFILES
from pokerbench.agents.strength_gates import (
    REFERENCE_OPPORTUNITIES_PER_SCENARIO,
    PolicyStrengthReference,
)
from pokerbench.engine.config import (
    BASELINE_GAME_CONFIG,
    GameConfig,
    config_hash,
)
from pokerbench.evaluation.calibration import (
    CALIBRATION_VERSION,
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    BehaviorCounts,
    CalibrationMetric,
    CalibrationMetricResult,
    CalibrationResult,
)
from pokerbench.evaluation.calibration_snapshots import (
    CALIBRATION_STRENGTH_SNAPSHOT_VERSION,
    CalibrationScenarioStrengthSnapshot,
    CalibrationStrengthSnapshotStore,
    generate_gate6_calibration_snapshots,
    policy_strength_reference_from_store,
    policy_validation_strengths_from_store,
)
from pokerbench.evaluation.policy_acceptance import (
    PolicyCalibrationAcceptance,
    PolicyValidationStrengths,
    StaticProfileGate6Result,
    evaluate_static_profile_gate6,
)


GATE6_CALIBRATION_REPORT_VERSION = "gate6-calibration-report-v1"


class Gate6CalibrationReportError(ValueError):
    """Base error for persisted Gate-6 reports."""


class Gate6CalibrationReportIntegrityError(Gate6CalibrationReportError):
    """Raised when persisted Gate-6 report content is invalid or inconsistent."""


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a SHA-256 hex digest") from exc


def _validate_baseline_game_config(game_config: GameConfig) -> None:
    if not isinstance(game_config, GameConfig):
        raise TypeError("game_config must be a GameConfig")
    if game_config != BASELINE_GAME_CONFIG:
        raise ValueError("Gate-6 execution requires BASELINE_GAME_CONFIG")


def _snapshot_key(
    phase: CalibrationPhase,
    scenario: CalibrationScenario,
) -> str:
    return f"{phase.value}/{scenario.value}"


def _expected_snapshot_keys() -> frozenset[str]:
    return frozenset(
        _snapshot_key(phase, scenario)
        for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION)
        for scenario in CalibrationScenario
    )


def _scenario_snapshot_map(
    snapshots: Sequence[CalibrationScenarioStrengthSnapshot],
    *,
    game_config: GameConfig,
) -> dict[tuple[CalibrationPhase, CalibrationScenario], CalibrationScenarioStrengthSnapshot]:
    expected_identity = {
        (phase, scenario)
        for phase in (CalibrationPhase.FIT, CalibrationPhase.VALIDATION)
        for scenario in CalibrationScenario
    }
    by_identity: dict[
        tuple[CalibrationPhase, CalibrationScenario],
        CalibrationScenarioStrengthSnapshot,
    ] = {}

    for snapshot in snapshots:
        if not isinstance(snapshot, CalibrationScenarioStrengthSnapshot):
            raise TypeError(
                "snapshots must contain CalibrationScenarioStrengthSnapshot values"
            )
        identity = (snapshot.phase, snapshot.scenario)
        if identity in by_identity:
            raise ValueError("Gate-6 report requires exactly all ten snapshot identities")
        if not snapshot.is_complete:
            raise ValueError("Gate-6 report cannot use incomplete calibration snapshots")
        if snapshot.game_config_hash != config_hash(game_config):
            raise ValueError("Calibration snapshot game-config hash mismatch")
        by_identity[identity] = snapshot

    if set(by_identity) != expected_identity:
        raise ValueError("Gate-6 report requires exactly all ten snapshot identities")
    return by_identity


def _reference_from_snapshot_map(
    snapshots: Mapping[
        tuple[CalibrationPhase, CalibrationScenario],
        CalibrationScenarioStrengthSnapshot,
    ],
) -> PolicyStrengthReference:
    def strengths(scenario: CalibrationScenario) -> tuple[int, ...]:
        return snapshots[(CalibrationPhase.FIT, scenario)].complete_strengths()

    return PolicyStrengthReference(
        open_btn=strengths(CalibrationScenario.OPEN_BTN),
        three_bet_bb=strengths(CalibrationScenario.THREE_BET_BB),
        cbet_btn=strengths(CalibrationScenario.CBET_BTN),
        fold_to_cbet_bb=strengths(CalibrationScenario.FOLD_TO_CBET_BB),
        postflop_aggression_btn=strengths(
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN
        ),
    )


def _validation_from_snapshot_map(
    snapshots: Mapping[
        tuple[CalibrationPhase, CalibrationScenario],
        CalibrationScenarioStrengthSnapshot,
    ],
) -> PolicyValidationStrengths:
    def strengths(scenario: CalibrationScenario) -> tuple[int, ...]:
        return snapshots[(CalibrationPhase.VALIDATION, scenario)].complete_strengths()

    return PolicyValidationStrengths(
        open_btn=strengths(CalibrationScenario.OPEN_BTN),
        three_bet_bb=strengths(CalibrationScenario.THREE_BET_BB),
        cbet_btn=strengths(CalibrationScenario.CBET_BTN),
        fold_to_cbet_bb=strengths(CalibrationScenario.FOLD_TO_CBET_BB),
        postflop_aggression_btn=strengths(
            CalibrationScenario.POSTFLOP_AGGRESSION_BTN
        ),
    )


def _common_acceptance_hash(
    gate6_result: StaticProfileGate6Result,
    *,
    field_name: str,
) -> str:
    values = {getattr(result, field_name) for result in gate6_result.results}
    if len(values) != 1:
        readable = field_name.replace("_", " ")
        raise ValueError(f"Gate-6 profile results do not share one {readable}")
    return next(iter(values))


def _validate_static_profile_identity(gate6_result: StaticProfileGate6Result) -> None:
    profiles_by_name = {
        profile.name.value: profile
        for profile in STATIC_PROFILES.values()
    }
    for acceptance in gate6_result.results:
        profile = profiles_by_name[acceptance.profile_name]
        if acceptance.profile_version != profile.version:
            raise ValueError("Gate-6 result uses an unsupported static-profile version")
        if acceptance.parameter_schema_version != profile.parameters.schema_version:
            raise ValueError("Gate-6 result uses an unsupported parameter-schema version")


@dataclass(frozen=True)
class Gate6CalibrationReport:
    """Immutable reproducibility report for one complete Gate-6 evaluation."""

    gate6_result: StaticProfileGate6Result
    snapshot_hashes: Mapping[str, str]
    game_config_hash: str
    report_version: str = GATE6_CALIBRATION_REPORT_VERSION
    snapshot_version: str = CALIBRATION_STRENGTH_SNAPSHOT_VERSION
    calibration_version: str = CALIBRATION_VERSION
    calibration_environment_version: str = CALIBRATION_ENVIRONMENT_VERSION
    opportunity_definition_version: str = OPPORTUNITY_DEFINITION_CONTRACT_VERSION
    fit_base_seed: int = FIT_BASE_SEED
    validation_base_seed: int = VALIDATION_BASE_SEED
    opportunities_per_metric: int = REFERENCE_OPPORTUNITIES_PER_SCENARIO

    def __post_init__(self) -> None:
        if not isinstance(self.gate6_result, StaticProfileGate6Result):
            raise TypeError("gate6_result must be a StaticProfileGate6Result")
        if self.report_version != GATE6_CALIBRATION_REPORT_VERSION:
            raise ValueError("Unsupported Gate-6 report version")
        if self.snapshot_version != CALIBRATION_STRENGTH_SNAPSHOT_VERSION:
            raise ValueError("Unsupported calibration snapshot version")
        if self.calibration_version != CALIBRATION_VERSION:
            raise ValueError("Unsupported calibration version")
        if self.calibration_environment_version != CALIBRATION_ENVIRONMENT_VERSION:
            raise ValueError("Unsupported calibration-environment version")
        if self.opportunity_definition_version != OPPORTUNITY_DEFINITION_CONTRACT_VERSION:
            raise ValueError("Unsupported opportunity-definition version")
        if self.fit_base_seed != FIT_BASE_SEED:
            raise ValueError("Unsupported fit base seed")
        if self.validation_base_seed != VALIDATION_BASE_SEED:
            raise ValueError("Unsupported validation base seed")
        if self.opportunities_per_metric != REFERENCE_OPPORTUNITIES_PER_SCENARIO:
            raise ValueError("Unsupported opportunities_per_metric")
        if self.game_config_hash != config_hash(BASELINE_GAME_CONFIG):
            raise ValueError("Gate-6 report game-config hash is not baseline")

        normalized_hashes = dict(self.snapshot_hashes)
        if set(normalized_hashes) != _expected_snapshot_keys():
            raise ValueError("snapshot_hashes must contain exactly all ten identities")
        for key, value in normalized_hashes.items():
            _validate_sha256(value, field_name=f"snapshot_hashes[{key!r}]")
        object.__setattr__(
            self,
            "snapshot_hashes",
            MappingProxyType(dict(sorted(normalized_hashes.items()))),
        )

        _validate_static_profile_identity(self.gate6_result)
        _common_acceptance_hash(
            self.gate6_result,
            field_name="reference_strength_hash",
        )
        _common_acceptance_hash(
            self.gate6_result,
            field_name="validation_strength_hash",
        )

    @classmethod
    def from_artifacts(
        cls,
        gate6_result: StaticProfileGate6Result,
        snapshots: Sequence[CalibrationScenarioStrengthSnapshot],
        *,
        game_config: GameConfig = BASELINE_GAME_CONFIG,
    ) -> "Gate6CalibrationReport":
        _validate_baseline_game_config(game_config)
        if not isinstance(gate6_result, StaticProfileGate6Result):
            raise TypeError("gate6_result must be a StaticProfileGate6Result")

        by_identity = _scenario_snapshot_map(snapshots, game_config=game_config)
        reference = _reference_from_snapshot_map(by_identity)
        validation = _validation_from_snapshot_map(by_identity)

        result_reference_hash = _common_acceptance_hash(
            gate6_result,
            field_name="reference_strength_hash",
        )
        result_validation_hash = _common_acceptance_hash(
            gate6_result,
            field_name="validation_strength_hash",
        )
        if result_reference_hash != reference.reference_strength_hash:
            raise ValueError(
                "Gate-6 result reference strength hash does not match fit snapshots"
            )
        if result_validation_hash != validation.validation_strength_hash:
            raise ValueError(
                "Gate-6 result validation strength hash does not match validation snapshots"
            )

        snapshot_hashes = {
            _snapshot_key(phase, scenario): snapshot.snapshot_hash
            for (phase, scenario), snapshot in by_identity.items()
        }
        return cls(
            gate6_result=gate6_result,
            snapshot_hashes=snapshot_hashes,
            game_config_hash=config_hash(game_config),
        )

    @property
    def passed(self) -> bool:
        return self.gate6_result.passed

    @property
    def gate6_hash(self) -> str:
        return self.gate6_result.gate6_hash

    @property
    def reference_strength_hash(self) -> str:
        return _common_acceptance_hash(
            self.gate6_result,
            field_name="reference_strength_hash",
        )

    @property
    def validation_strength_hash(self) -> str:
        return _common_acceptance_hash(
            self.gate6_result,
            field_name="validation_strength_hash",
        )

    @property
    def profile_acceptance_hashes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                result.profile_name: result.acceptance_hash
                for result in self.gate6_result.results
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": self.report_version,
            "snapshot_version": self.snapshot_version,
            "calibration_version": self.calibration_version,
            "calibration_environment_version": self.calibration_environment_version,
            "opportunity_definition_version": self.opportunity_definition_version,
            "fit_base_seed": self.fit_base_seed,
            "validation_base_seed": self.validation_base_seed,
            "opportunities_per_metric": self.opportunities_per_metric,
            "game_config_hash": self.game_config_hash,
            "passed": self.passed,
            "reference_strength_hash": self.reference_strength_hash,
            "validation_strength_hash": self.validation_strength_hash,
            "gate6_hash": self.gate6_hash,
            "snapshot_hashes": dict(self.snapshot_hashes),
            "profile_acceptance_hashes": dict(self.profile_acceptance_hashes),
            "gate6_result": self.gate6_result.to_dict(),
        }

    @property
    def report_hash(self) -> str:
        return _sha256_payload(self.to_dict())


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Gate6CalibrationReportIntegrityError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(payload) != expected:
        raise Gate6CalibrationReportIntegrityError(f"{name} has unexpected fields")


def _parse_behavior_counts(payload: Any) -> BehaviorCounts:
    raw = _require_object(payload, name="behavior_counts")
    expected = {
        "vpip_opportunities",
        "vpip_events",
        "pfr_opportunities",
        "pfr_events",
        "three_bet_opportunities",
        "three_bet_events",
        "cbet_opportunities",
        "cbet_events",
        "postflop_aggressive_actions",
        "postflop_passive_actions",
        "fold_to_cbet_opportunities",
        "fold_to_cbet_events",
        "opportunity_definition_version",
    }
    _require_exact_keys(raw, expected, name="behavior_counts")
    try:
        return BehaviorCounts(**raw)
    except (TypeError, ValueError) as exc:
        raise Gate6CalibrationReportIntegrityError(
            "behavior_counts payload is invalid"
        ) from exc


def _parse_metric_result(payload: Any) -> CalibrationMetricResult:
    raw = _require_object(payload, name="calibration metric")
    expected = {
        "metric",
        "target",
        "observed",
        "tolerance",
        "numerator",
        "denominator",
        "required",
        "deviation",
        "passed",
    }
    _require_exact_keys(raw, expected, name="calibration metric")
    try:
        return CalibrationMetricResult(
            metric=CalibrationMetric(raw["metric"]),
            target=raw["target"],
            observed=raw["observed"],
            tolerance=raw["tolerance"],
            numerator=raw["numerator"],
            denominator=raw["denominator"],
            required=raw["required"],
            deviation=raw["deviation"],
            passed=raw["passed"],
        )
    except (TypeError, ValueError) as exc:
        raise Gate6CalibrationReportIntegrityError(
            "calibration metric payload is invalid"
        ) from exc


def _parse_calibration_result(payload: Any) -> CalibrationResult:
    raw = _require_object(payload, name="calibration_result")
    expected = {
        "profile_name",
        "profile_version",
        "parameter_schema_version",
        "calibration_version",
        "opportunity_definition_version",
        "tolerance_version",
        "is_calibrated",
        "metrics",
    }
    _require_exact_keys(raw, expected, name="calibration_result")
    metrics_raw = raw["metrics"]
    if not isinstance(metrics_raw, list):
        raise Gate6CalibrationReportIntegrityError(
            "calibration_result.metrics must be a JSON array"
        )
    try:
        return CalibrationResult(
            profile_name=raw["profile_name"],
            profile_version=raw["profile_version"],
            parameter_schema_version=raw["parameter_schema_version"],
            calibration_version=raw["calibration_version"],
            opportunity_definition_version=raw["opportunity_definition_version"],
            tolerance_version=raw["tolerance_version"],
            metrics=tuple(_parse_metric_result(item) for item in metrics_raw),
            is_calibrated=raw["is_calibrated"],
        )
    except (TypeError, ValueError) as exc:
        raise Gate6CalibrationReportIntegrityError(
            "calibration_result payload is invalid"
        ) from exc


def _parse_acceptance(payload: Any) -> PolicyCalibrationAcceptance:
    raw = _require_object(payload, name="profile acceptance")
    expected = {
        "calibration_version",
        "calibration_environment_version",
        "fit_base_seed",
        "validation_base_seed",
        "opportunities_per_metric",
        "profile_name",
        "profile_version",
        "parameter_schema_version",
        "reference_strength_hash",
        "gate_map_hash",
        "validation_strength_hash",
        "designated_opportunities",
        "behavior_counts",
        "calibration_result",
    }
    _require_exact_keys(raw, expected, name="profile acceptance")
    designated = raw["designated_opportunities"]
    if not isinstance(designated, dict):
        raise Gate6CalibrationReportIntegrityError(
            "designated_opportunities must be a JSON object"
        )
    try:
        return PolicyCalibrationAcceptance(
            profile_name=raw["profile_name"],
            profile_version=raw["profile_version"],
            parameter_schema_version=raw["parameter_schema_version"],
            reference_strength_hash=raw["reference_strength_hash"],
            gate_map_hash=raw["gate_map_hash"],
            validation_strength_hash=raw["validation_strength_hash"],
            behavior_counts=_parse_behavior_counts(raw["behavior_counts"]),
            designated_opportunities=designated,
            calibration_result=_parse_calibration_result(raw["calibration_result"]),
            calibration_version=raw["calibration_version"],
            calibration_environment_version=raw["calibration_environment_version"],
            fit_base_seed=raw["fit_base_seed"],
            validation_base_seed=raw["validation_base_seed"],
            opportunities_per_metric=raw["opportunities_per_metric"],
        )
    except (TypeError, ValueError) as exc:
        raise Gate6CalibrationReportIntegrityError(
            "profile acceptance payload is invalid"
        ) from exc


def parse_policy_calibration_acceptance_payload(
    payload: Any,
) -> PolicyCalibrationAcceptance:
    """Parse one persisted calibration-acceptance payload with full validation."""
    return _parse_acceptance(payload)


def _parse_gate6_result(payload: Any) -> StaticProfileGate6Result:
    raw = _require_object(payload, name="gate6_result")
    expected = {"calibration_version", "passed", "results"}
    _require_exact_keys(raw, expected, name="gate6_result")
    results_raw = raw["results"]
    if not isinstance(results_raw, list):
        raise Gate6CalibrationReportIntegrityError(
            "gate6_result.results must be a JSON array"
        )
    try:
        result = StaticProfileGate6Result(
            results=tuple(_parse_acceptance(item) for item in results_raw),
            calibration_version=raw["calibration_version"],
        )
    except (TypeError, ValueError) as exc:
        raise Gate6CalibrationReportIntegrityError(
            "gate6_result payload is invalid"
        ) from exc
    if result.passed is not raw["passed"]:
        raise Gate6CalibrationReportIntegrityError(
            "gate6_result.passed is inconsistent with profile results"
        )
    return result


def save_gate6_calibration_report(
    report: Gate6CalibrationReport,
    path: str | Path,
) -> Path:
    if not isinstance(report, Gate6CalibrationReport):
        raise TypeError("report must be a Gate6CalibrationReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "report": report.to_dict(),
        "report_hash": report.report_hash,
    }
    serialized = _canonical_json(payload) + "\n"

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_gate6_calibration_report(path: str | Path) -> Gate6CalibrationReport:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Gate6CalibrationReportIntegrityError(
            f"Invalid Gate-6 report JSON: {source}"
        ) from exc

    outer = _require_object(payload, name="Gate-6 report file")
    _require_exact_keys(outer, {"report", "report_hash"}, name="Gate-6 report file")
    raw_report = _require_object(outer["report"], name="report")
    stored_hash = outer["report_hash"]
    try:
        _validate_sha256(stored_hash, field_name="report_hash")
    except (TypeError, ValueError) as exc:
        raise Gate6CalibrationReportIntegrityError("Invalid Gate-6 report hash") from exc
    if _sha256_payload(raw_report) != stored_hash:
        raise Gate6CalibrationReportIntegrityError(
            "Gate-6 report canonical hash mismatch"
        )

    expected_report_keys = {
        "report_version",
        "snapshot_version",
        "calibration_version",
        "calibration_environment_version",
        "opportunity_definition_version",
        "fit_base_seed",
        "validation_base_seed",
        "opportunities_per_metric",
        "game_config_hash",
        "passed",
        "reference_strength_hash",
        "validation_strength_hash",
        "gate6_hash",
        "snapshot_hashes",
        "profile_acceptance_hashes",
        "gate6_result",
    }
    _require_exact_keys(raw_report, expected_report_keys, name="report")
    snapshot_hashes = raw_report["snapshot_hashes"]
    if not isinstance(snapshot_hashes, dict):
        raise Gate6CalibrationReportIntegrityError(
            "snapshot_hashes must be a JSON object"
        )

    try:
        report = Gate6CalibrationReport(
            gate6_result=_parse_gate6_result(raw_report["gate6_result"]),
            snapshot_hashes=snapshot_hashes,
            game_config_hash=raw_report["game_config_hash"],
            report_version=raw_report["report_version"],
            snapshot_version=raw_report["snapshot_version"],
            calibration_version=raw_report["calibration_version"],
            calibration_environment_version=raw_report[
                "calibration_environment_version"
            ],
            opportunity_definition_version=raw_report[
                "opportunity_definition_version"
            ],
            fit_base_seed=raw_report["fit_base_seed"],
            validation_base_seed=raw_report["validation_base_seed"],
            opportunities_per_metric=raw_report["opportunities_per_metric"],
        )
    except (TypeError, ValueError) as exc:
        raise Gate6CalibrationReportIntegrityError(
            "Gate-6 report payload is invalid"
        ) from exc

    if report.to_dict() != raw_report:
        raise Gate6CalibrationReportIntegrityError(
            "Gate-6 report derived fields are inconsistent"
        )
    if report.report_hash != stored_hash:
        raise Gate6CalibrationReportIntegrityError(
            "Gate-6 report typed reconstruction hash mismatch"
        )
    return report


_METRIC_LABELS = {
    CalibrationMetric.VPIP: "VPIP",
    CalibrationMetric.PFR: "PFR",
    CalibrationMetric.THREE_BET: "3BetFreq",
    CalibrationMetric.CBET: "CBetFreq",
    CalibrationMetric.AGGRESSION: "Aggression",
    CalibrationMetric.FOLD_TO_CBET: "FoldToCBet",
}


def _format_optional(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def render_gate6_calibration_report_text(report: Gate6CalibrationReport) -> str:
    if not isinstance(report, Gate6CalibrationReport):
        raise TypeError("report must be a Gate6CalibrationReport")

    lines = [
        f"Gate 6: {'PASS' if report.passed else 'FAIL'}",
        f"Report version: {report.report_version}",
        f"Report hash: {report.report_hash}",
        f"Gate-6 hash: {report.gate6_hash}",
        f"Reference strength hash: {report.reference_strength_hash}",
        f"Validation strength hash: {report.validation_strength_hash}",
        "",
    ]

    for acceptance in report.gate6_result.results:
        lines.append(
            f"{acceptance.profile_name}: "
            f"{'PASS' if acceptance.is_calibrated else 'FAIL'}"
        )
        lines.append(f"  Acceptance hash: {acceptance.acceptance_hash}")
        for metric in acceptance.calibration_result.metrics:
            label = _METRIC_LABELS[metric.metric]
            required = "required" if metric.required else "don't-care"
            status = (
                "PASS"
                if metric.passed is True
                else "FAIL"
                if metric.passed is False
                else "N/A"
            )
            lines.append(
                "  "
                f"{label}: target={_format_optional(metric.target)} "
                f"observed={_format_optional(metric.observed)} "
                f"tolerance={metric.tolerance:.6f} "
                f"numerator={metric.numerator} denominator={metric.denominator} "
                f"{required} {status}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def execute_gate6_calibration(
    snapshot_root: str | Path,
    report_path: str | Path,
    *,
    workers: int = 1,
    checkpoint_every: int = 25,
    game_config: GameConfig = BASELINE_GAME_CONFIG,
) -> Gate6CalibrationReport:
    """Run/resume the frozen Gate-6 workflow synchronously and persist its report."""
    _validate_baseline_game_config(game_config)
    destination = Path(report_path)
    existing = (
        load_gate6_calibration_report(destination)
        if destination.is_file()
        else None
    )

    store = CalibrationStrengthSnapshotStore(snapshot_root)
    snapshots = generate_gate6_calibration_snapshots(
        store,
        workers=workers,
        checkpoint_every=checkpoint_every,
        game_config=game_config,
    )
    reference = policy_strength_reference_from_store(store)
    validation = policy_validation_strengths_from_store(store)
    gate6_result = evaluate_static_profile_gate6(
        reference,
        validation,
        game_config=game_config,
    )
    report = Gate6CalibrationReport.from_artifacts(
        gate6_result,
        snapshots,
        game_config=game_config,
    )

    if existing is not None:
        if existing.report_hash != report.report_hash:
            raise Gate6CalibrationReportIntegrityError(
                "Existing Gate-6 report contains a different deterministic result"
            )
        return existing

    save_gate6_calibration_report(report, destination)
    return report
