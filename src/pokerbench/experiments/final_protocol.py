"""Frozen Phase-1 final baseline experiment loading and validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

from pokerbench.agents.hand_strength_baseline import HAND_STRENGTH_BASELINE_VERSION
from pokerbench.agents.profiles import STATIC_PROFILE_VERSION
from pokerbench.agents.random_baseline import RANDOM_BASELINE_VERSION
from pokerbench.agents.strength_gates import PARAMETRIC_POLICY_VERSION
from pokerbench.engine.config import BASELINE_GAME_CONFIG, config_hash
from pokerbench.evaluation.population import (
    POPULATION_EVALUATION_VERSION,
    POPULATION_WEIGHT_SCALE,
    DiscretePopulationDefinition,
)
from pokerbench.experiments.frozen_protocol import canonical_file_bytes
from pokerbench.runner import IllegalActionPolicy, StatisticalUnit


PHASE1_BASELINE_PROTOCOL_FILE_VERSION = "phase1-baseline-experiment-file-v1"
PHASE1_BASELINE_PROTOCOL_VERSION = "phase1-baseline-experiment-protocol-v1"
PHASE1_BASELINE_PROTOCOL_ID = "phase1-baseline-experiment-v1"
PHASE1_FINAL_SEED_SCHEDULE_VERSION = "phase1-final-seeds-v1"
PHASE1_FINAL_PRECISION_PLAN_VERSION = "population-mean-normal-precision-plan-v1"
PHASE1_FINAL_RNG_CONTRACT = "event-indexed-crn-v1"
PHASE1_FINAL_CONFIDENCE_LEVEL = 0.95
PHASE1_FINAL_TARGET_HALF_WIDTH_BB100 = 50.0
PHASE1_PROJECT_ILLUSTRATIVE_HALF_WIDTH_BB100 = 1.0
PHASE1_FINAL_Z = NormalDist().inv_cdf(0.975)
PHASE1_FINAL_SEED_START = 6_200_000
PHASE1_PILOT_SEED_START = 6_100_000
PHASE1_PILOT_SEED_COUNT = 100
PHASE1_PILOT_PROTOCOL_HASH = (
    "cc589fc65610bca308d8ed739efab848fff8477e88c20043b045d5ad3ee6c0a9"
)
PHASE1_PILOT_REPORT_HASH = (
    "088977f01dff06dc1d64146be35ab2b04f8246e30f7c0130e398c29d6cf24fd1"
)
PHASE1_GATE6_REPORT_HASH = (
    "ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a"
)


class FinalProtocolError(ValueError):
    """Raised when the frozen Phase-1 final protocol is inconsistent."""


@dataclass(frozen=True)
class FinalAgentPlan:
    """One final hero agent and its fixed equal-per-profile allocation."""

    agent_id: str
    agent_version: str
    paired_units_per_profile: int
    seed_start: int
    seed_count: int
    raw_required_units_per_profile: float
    planned_half_width_bb100: float

    @property
    def seed_end_inclusive(self) -> int:
        return self.seed_start + self.seed_count - 1

    @property
    def master_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))


@dataclass(frozen=True)
class FinalPilotVarianceCell:
    """Pilot SD evidence used solely for fixed final sample-size planning."""

    hero_id: str
    opponent_profile_id: str
    std_difference_bb100: float
    valid_units: int


@dataclass(frozen=True)
class Phase1BaselineExperimentProtocol:
    """Validated immutable final Phase-1 baseline experiment design."""

    protocol_id: str
    protocol_version: str
    protocol_hash: str
    status: str
    purpose: str
    statistical_unit: str
    game_config_hash: str
    primary_population_id: str
    primary_population_hash: str
    secondary_population_ids: tuple[str, ...]
    opponent_profile_ids: tuple[str, ...]
    agent_plans: tuple[FinalAgentPlan, ...]
    pilot_variance_cells: tuple[FinalPilotVarianceCell, ...]
    pilot_protocol_hash: str
    pilot_report_hash: str
    gate6_report_hash: str
    confidence_level: float
    target_half_width_bb100: float
    illustrative_project_half_width_bb100: float
    planned_independent_units: int
    planned_simulated_engine_hands: int
    engine_hands_per_paired_unit: int
    seed_schedule_version: str
    replacement_seed_policy: str
    illegal_action_policy: str


@dataclass(frozen=True)
class FinalJob:
    """One frozen hero/profile/seed final experiment unit."""

    cell_id: str
    hero_id: str
    hero_version: str
    opponent_profile_id: str
    opponent_profile_version: str
    master_seed: int
    paired_units: int = 1
    simulated_engine_hands: int = 2


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_canonical(value: Any) -> str:
    """Return SHA-256 over canonical compact JSON without a trailing newline."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalProtocolError(f"{field_name} must be a JSON object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalProtocolError(f"{field_name} must be a non-empty string")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise FinalProtocolError(f"{field_name} must be boolean")
    return value


def _require_int(value: Any, field_name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalProtocolError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise FinalProtocolError(f"{field_name} must be positive")
    return value


def _require_float(value: Any, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalProtocolError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FinalProtocolError(f"{field_name} must be finite")
    if positive and result <= 0.0:
        raise FinalProtocolError(f"{field_name} must be positive")
    return result


def _require_sha256(value: Any, field_name: str) -> str:
    digest = _require_string(value, field_name).lower()
    if len(digest) != 64:
        raise FinalProtocolError(f"{field_name} must be a SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise FinalProtocolError(f"{field_name} must be hexadecimal") from exc
    return digest


def _require_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise FinalProtocolError(f"{field_name} must be a non-empty JSON array")
    result = tuple(_require_string(item, field_name) for item in value)
    if len(set(result)) != len(result):
        raise FinalProtocolError(f"{field_name} must contain unique values")
    return result


def _population_reference(
    raw: Any,
    *,
    field_name: str,
    populations: Mapping[str, DiscretePopulationDefinition],
) -> tuple[str, str, str]:
    value = _require_mapping(raw, field_name)
    population_id = _require_string(value.get("population_id"), f"{field_name}.population_id")
    version = _require_string(value.get("version"), f"{field_name}.version")
    population_hash = _require_sha256(value.get("population_hash"), f"{field_name}.population_hash")
    population = populations.get(population_id)
    if population is None:
        raise FinalProtocolError(f"{field_name} has no supplied frozen population")
    if population.version != version or population.population_hash != population_hash:
        raise FinalProtocolError(f"{field_name} does not match supplied frozen population")
    return population_id, version, population_hash


def _variance_term(
    cells: tuple[FinalPilotVarianceCell, ...],
    *,
    hero_id: str,
    primary_population: DiscretePopulationDefinition,
) -> float:
    by_profile = {
        cell.opponent_profile_id: cell.std_difference_bb100
        for cell in cells
        if cell.hero_id == hero_id
    }
    if set(by_profile) != {profile_id for profile_id, _ in primary_population.weights_ppm}:
        raise FinalProtocolError(f"Pilot variance evidence is incomplete for {hero_id}")
    return sum(
        (weight_ppm / POPULATION_WEIGHT_SCALE) ** 2
        * (by_profile[profile_id] / 2.0) ** 2
        for profile_id, weight_ppm in primary_population.weights_ppm
    )


def precision_plan(
    cells: tuple[FinalPilotVarianceCell, ...],
    *,
    hero_id: str,
    primary_population: DiscretePopulationDefinition,
    target_half_width_bb100: float = PHASE1_FINAL_TARGET_HALF_WIDTH_BB100,
    z_value: float = PHASE1_FINAL_Z,
) -> tuple[float, int, float]:
    """Return raw n, fixed ceil(n), and planned half-width for one hero."""
    if target_half_width_bb100 <= 0.0 or not math.isfinite(target_half_width_bb100):
        raise FinalProtocolError("target_half_width_bb100 must be finite and positive")
    if z_value <= 0.0 or not math.isfinite(z_value):
        raise FinalProtocolError("z_value must be finite and positive")
    variance_term = _variance_term(
        cells,
        hero_id=hero_id,
        primary_population=primary_population,
    )
    raw_n = (z_value * math.sqrt(variance_term) / target_half_width_bb100) ** 2
    fixed_n = math.ceil(raw_n)
    planned_half_width = z_value * math.sqrt(variance_term / fixed_n)
    return raw_n, fixed_n, planned_half_width


def load_phase1_baseline_protocol(
    path: str | Path,
    *,
    populations: Mapping[str, DiscretePopulationDefinition],
) -> Phase1BaselineExperimentProtocol:
    """Load and strictly validate the frozen Phase-1 final baseline protocol."""
    source = Path(path)
    raw_bytes = source.read_bytes()
    raw = json.loads(raw_bytes)
    if raw_bytes != canonical_file_bytes(raw):
        raise FinalProtocolError("Final protocol file is not canonical UTF-8/LF JSON")
    envelope = _require_mapping(raw, "final protocol file")
    if envelope.get("file_version") != PHASE1_BASELINE_PROTOCOL_FILE_VERSION:
        raise FinalProtocolError("Unsupported final protocol file_version")
    stored_hash = _require_sha256(envelope.get("protocol_hash"), "protocol_hash")
    payload = _require_mapping(envelope.get("protocol"), "protocol")
    if sha256_canonical(payload) != stored_hash:
        raise FinalProtocolError("protocol_hash does not match protocol payload")

    protocol_id = _require_string(payload.get("protocol_id"), "protocol_id")
    protocol_version = _require_string(payload.get("protocol_version"), "protocol_version")
    status = _require_string(payload.get("status"), "status")
    purpose = _require_string(payload.get("purpose"), "purpose")
    if (
        protocol_id != PHASE1_BASELINE_PROTOCOL_ID
        or protocol_version != PHASE1_BASELINE_PROTOCOL_VERSION
        or status != "frozen"
        or purpose != "final_phase1_static_baseline_measurement"
    ):
        raise FinalProtocolError("Final Phase-1 protocol identity/status/purpose mismatch")

    scientific_use = _require_mapping(payload.get("scientific_use"), "scientific_use")
    if not _require_bool(scientific_use.get("final_experiment_data"), "scientific_use.final_experiment_data"):
        raise FinalProtocolError("Final protocol must produce final experiment data")
    if _require_bool(scientific_use.get("eligible_for_confirmatory_analysis"), "scientific_use.eligible_for_confirmatory_analysis"):
        raise FinalProtocolError("Phase-1 v1 is not a confirmatory hypothesis test")
    if _require_bool(scientific_use.get("hypothesis_test_planned"), "scientific_use.hypothesis_test_planned"):
        raise FinalProtocolError("No Phase-1 confirmatory hypothesis test is frozen")
    if _require_bool(scientific_use.get("optional_stopping"), "scientific_use.optional_stopping"):
        raise FinalProtocolError("Ordinary optional stopping is prohibited")
    if _require_bool(scientific_use.get("pilot_data_included_in_final_estimate"), "scientific_use.pilot_data_included_in_final_estimate"):
        raise FinalProtocolError("Pilot observations must be excluded from final estimates")

    power = _require_mapping(payload.get("power"), "power")
    if power.get("alpha") is not None or power.get("beta") is not None or power.get("minimum_relevant_difference_bb100") is not None:
        raise FinalProtocolError("Power/effect-size fields must remain null without a hypothesis test")
    if power.get("reason") != "no_confirmatory_phase1_hypothesis_test_is_frozen":
        raise FinalProtocolError("Power-null rationale must remain explicit")

    game = _require_mapping(payload.get("game"), "game")
    game_hash = _require_sha256(game.get("game_config_hash"), "game.game_config_hash")
    if game.get("abstraction_version") != BASELINE_GAME_CONFIG.abstraction_version:
        raise FinalProtocolError("Final game abstraction does not match BASELINE_GAME_CONFIG")
    if game_hash != config_hash(BASELINE_GAME_CONFIG):
        raise FinalProtocolError("Final game_config_hash does not match BASELINE_GAME_CONFIG")

    statistical_unit = _require_string(payload.get("statistical_unit"), "statistical_unit")
    if statistical_unit != StatisticalUnit.PAIRED_DEAL.value:
        raise FinalProtocolError("Final static baseline requires paired_deal statistical units")

    opponents = _require_mapping(payload.get("opponents"), "opponents")
    opponent_profile_ids = _require_string_tuple(opponents.get("profile_ids"), "opponents.profile_ids")
    if opponent_profile_ids != ("nit", "tag", "lag", "calling_station", "maniac"):
        raise FinalProtocolError("Final opponents must be the five frozen static profiles")
    if opponents.get("static_profile_version") != STATIC_PROFILE_VERSION:
        raise FinalProtocolError("Static profile version mismatch")
    if opponents.get("parametric_policy_version") != PARAMETRIC_POLICY_VERSION:
        raise FinalProtocolError("Parametric policy version mismatch")
    gate6_report_hash = _require_sha256(opponents.get("gate6_report_hash"), "opponents.gate6_report_hash")
    if gate6_report_hash != PHASE1_GATE6_REPORT_HASH:
        raise FinalProtocolError("Gate-6 report hash must equal the pilot-bound passing report")

    population_block = _require_mapping(payload.get("populations"), "populations")
    primary_id, _, primary_hash = _population_reference(
        population_block.get("primary"), field_name="populations.primary", populations=populations
    )
    primary_population = populations[primary_id]
    if primary_id != "baseline-synthetic-static-profiles":
        raise FinalProtocolError("Baseline Synthetic must remain the primary Phase-1 population")
    raw_secondary = population_block.get("secondary_descriptive")
    if not isinstance(raw_secondary, list) or not raw_secondary:
        raise FinalProtocolError("populations.secondary_descriptive must be non-empty")
    secondary_ids = tuple(
        _population_reference(
            item,
            field_name=f"populations.secondary_descriptive[{index}]",
            populations=populations,
        )[0]
        for index, item in enumerate(raw_secondary)
    )
    if secondary_ids != ("balanced-static-profiles",):
        raise FinalProtocolError("Balanced must remain the only secondary descriptive population")

    planning = _require_mapping(payload.get("pilot_planning_basis"), "pilot_planning_basis")
    pilot_protocol_hash = _require_sha256(planning.get("pilot_protocol_hash"), "pilot_planning_basis.pilot_protocol_hash")
    pilot_report_hash = _require_sha256(planning.get("pilot_report_hash"), "pilot_planning_basis.pilot_report_hash")
    pilot_gate6_hash = _require_sha256(planning.get("gate6_report_hash"), "pilot_planning_basis.gate6_report_hash")
    if planning.get("pilot_protocol_id") != "phase1-pilot-v1":
        raise FinalProtocolError("Pilot protocol id mismatch")
    if pilot_protocol_hash != PHASE1_PILOT_PROTOCOL_HASH or pilot_report_hash != PHASE1_PILOT_REPORT_HASH:
        raise FinalProtocolError("Final design must remain bound to the completed frozen pilot")
    if pilot_gate6_hash != gate6_report_hash:
        raise FinalProtocolError("Final and pilot Gate-6 report hashes must match")
    if (
        planning.get("completed_jobs") != 1000
        or planning.get("valid_units") != 1000
        or planning.get("failed_units") != 0
        or planning.get("recorded_engine_hands") != 2000
    ):
        raise FinalProtocolError("Frozen pilot completion summary mismatch")
    if planning.get("variance_estimator") != "sample-sd-paired-difference-bb100-ddof1-v1":
        raise FinalProtocolError("Pilot variance estimator mismatch")

    raw_cells = planning.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 10:
        raise FinalProtocolError("pilot_planning_basis.cells must contain exactly ten cells")
    pilot_cells: list[FinalPilotVarianceCell] = []
    for index, raw_cell in enumerate(raw_cells):
        cell = _require_mapping(raw_cell, f"pilot_planning_basis.cells[{index}]")
        pilot_cells.append(
            FinalPilotVarianceCell(
                hero_id=_require_string(cell.get("hero_id"), f"pilot_planning_basis.cells[{index}].hero_id"),
                opponent_profile_id=_require_string(cell.get("opponent_profile_id"), f"pilot_planning_basis.cells[{index}].opponent_profile_id"),
                std_difference_bb100=_require_float(cell.get("std_difference_bb100"), f"pilot_planning_basis.cells[{index}].std_difference_bb100", positive=True),
                valid_units=_require_int(cell.get("valid_units"), f"pilot_planning_basis.cells[{index}].valid_units", positive=True),
            )
        )
    cells_tuple = tuple(pilot_cells)
    expected_cell_keys = {
        (hero_id, profile_id)
        for hero_id in ("random", "hand_strength")
        for profile_id in opponent_profile_ids
    }
    if {(cell.hero_id, cell.opponent_profile_id) for cell in cells_tuple} != expected_cell_keys:
        raise FinalProtocolError("Pilot variance cell identities are incomplete or duplicated")
    if any(cell.valid_units != 100 for cell in cells_tuple):
        raise FinalProtocolError("Each pilot variance cell must retain all 100 frozen valid units")

    precision = _require_mapping(payload.get("precision"), "precision")
    if precision.get("plan_version") != PHASE1_FINAL_PRECISION_PLAN_VERSION:
        raise FinalProtocolError("Unsupported precision plan version")
    confidence_level = _require_float(precision.get("confidence_level"), "precision.confidence_level", positive=True)
    target_half_width = _require_float(precision.get("target_half_width_bb100"), "precision.target_half_width_bb100", positive=True)
    z_value = _require_float(precision.get("normal_z_value"), "precision.normal_z_value", positive=True)
    if confidence_level != PHASE1_FINAL_CONFIDENCE_LEVEL or target_half_width != PHASE1_FINAL_TARGET_HALF_WIDTH_BB100:
        raise FinalProtocolError("Final precision target must remain frozen at 95% and ±50 BB/100")
    if not math.isclose(z_value, PHASE1_FINAL_Z, rel_tol=0.0, abs_tol=1e-15):
        raise FinalProtocolError("precision.normal_z_value does not match NormalDist(0.975)")
    if precision.get("primary_estimand") != "baseline_synthetic_population_mean_ev_bb100":
        raise FinalProtocolError("Primary precision estimand mismatch")
    if not _require_bool(
        precision.get("equal_units_per_profile_within_hero"),
        "precision.equal_units_per_profile_within_hero",
    ):
        raise FinalProtocolError("Equal per-profile allocation within hero must remain frozen")
    if precision.get("hero_sd_conversion") != "hero-ev-sd-is-paired-difference-sd-divided-by-2-v1":
        raise FinalProtocolError("Hero EV variance conversion mismatch")

    disclosure = _require_mapping(precision.get("disclosure"), "precision.disclosure")
    illustrative = _require_float(
        disclosure.get("illustrative_project_example_half_width_bb100"),
        "precision.disclosure.illustrative_project_example_half_width_bb100",
        positive=True,
    )
    if illustrative != PHASE1_PROJECT_ILLUSTRATIVE_HALF_WIDTH_BB100:
        raise FinalProtocolError("Illustrative ±1 BB/100 project example must remain documented")
    if not _require_bool(disclosure.get("deviation_from_illustrative_example"), "precision.disclosure.deviation_from_illustrative_example"):
        raise FinalProtocolError("Coarser final precision choice must remain explicitly disclosed")
    if disclosure.get("selection_basis") != "completed_phase1_pilot_variance_and_runtime_before_final_data":
        raise FinalProtocolError("Precision selection basis must remain pilot-before-final-data")

    raw_heroes = payload.get("hero_agents")
    if not isinstance(raw_heroes, list) or len(raw_heroes) != 2:
        raise FinalProtocolError("hero_agents must contain exactly two final baselines")
    expected_versions = {
        "random": RANDOM_BASELINE_VERSION,
        "hand_strength": HAND_STRENGTH_BASELINE_VERSION,
    }
    plans: list[FinalAgentPlan] = []
    for index, raw_hero in enumerate(raw_heroes):
        hero = _require_mapping(raw_hero, f"hero_agents[{index}]")
        hero_id = _require_string(hero.get("agent_id"), f"hero_agents[{index}].agent_id")
        hero_version = _require_string(hero.get("agent_version"), f"hero_agents[{index}].agent_version")
        if expected_versions.get(hero_id) != hero_version or _require_bool(hero.get("adaptive"), f"hero_agents[{index}].adaptive"):
            raise FinalProtocolError("Final hero baseline identity/version/adaptivity mismatch")
        n = _require_int(hero.get("paired_units_per_profile"), f"hero_agents[{index}].paired_units_per_profile", positive=True)
        seed_start = _require_int(hero.get("seed_start"), f"hero_agents[{index}].seed_start")
        seed_count = _require_int(hero.get("seed_count"), f"hero_agents[{index}].seed_count", positive=True)
        raw_required = _require_float(hero.get("raw_required_units_per_profile"), f"hero_agents[{index}].raw_required_units_per_profile", positive=True)
        planned_half_width = _require_float(hero.get("planned_half_width_bb100"), f"hero_agents[{index}].planned_half_width_bb100", positive=True)
        calc_raw, calc_n, calc_half_width = precision_plan(
            cells_tuple,
            hero_id=hero_id,
            primary_population=primary_population,
            target_half_width_bb100=target_half_width,
            z_value=z_value,
        )
        if n != calc_n or seed_count != calc_n:
            raise FinalProtocolError(f"{hero_id} fixed sample size is not ceil(pilot precision requirement)")
        if seed_start != PHASE1_FINAL_SEED_START:
            raise FinalProtocolError("Both hero schedules must use the frozen common final seed prefix")
        if not math.isclose(raw_required, calc_raw, rel_tol=0.0, abs_tol=1e-12):
            raise FinalProtocolError(f"{hero_id} raw_required_units_per_profile mismatch")
        if not math.isclose(planned_half_width, calc_half_width, rel_tol=0.0, abs_tol=1e-12):
            raise FinalProtocolError(f"{hero_id} planned_half_width_bb100 mismatch")
        if planned_half_width > target_half_width:
            raise FinalProtocolError(f"{hero_id} planned half-width exceeds the frozen target")
        plans.append(
            FinalAgentPlan(
                agent_id=hero_id,
                agent_version=hero_version,
                paired_units_per_profile=n,
                seed_start=seed_start,
                seed_count=seed_count,
                raw_required_units_per_profile=raw_required,
                planned_half_width_bb100=planned_half_width,
            )
        )
    plans_tuple = tuple(plans)
    if tuple(plan.agent_id for plan in plans_tuple) != ("random", "hand_strength"):
        raise FinalProtocolError("Final hero plan ordering is frozen")

    rng = _require_mapping(payload.get("rng"), "rng")
    if (
        rng.get("rng_contract") != PHASE1_FINAL_RNG_CONTRACT
        or rng.get("seed_schedule_version") != PHASE1_FINAL_SEED_SCHEDULE_VERSION
        or rng.get("crn_mode") != "identical_realisations"
        or rng.get("deal_namespace") != "DEAL_SCHEDULE"
        or rng.get("bot_a_namespace") != "RNG_BOT_A"
        or rng.get("bot_b_namespace") != "RNG_BOT_B"
        or rng.get("agent_event_key") != "(bot_id,hand_id,decision_index,purpose)"
        or not _require_bool(rng.get("reuse_same_seed_schedule_across_opponent_cells"), "rng.reuse_same_seed_schedule_across_opponent_cells")
        or not _require_bool(rng.get("reuse_common_seed_prefix_across_heroes"), "rng.reuse_common_seed_prefix_across_heroes")
        or not _require_bool(rng.get("pilot_seed_overlap_forbidden"), "rng.pilot_seed_overlap_forbidden")
    ):
        raise FinalProtocolError("Final RNG/CRN seed contract mismatch")
    pilot_seed_start = _require_int(rng.get("pilot_seed_start"), "rng.pilot_seed_start")
    pilot_seed_count = _require_int(rng.get("pilot_seed_count"), "rng.pilot_seed_count", positive=True)
    if pilot_seed_start != PHASE1_PILOT_SEED_START or pilot_seed_count != PHASE1_PILOT_SEED_COUNT:
        raise FinalProtocolError("Pilot seed exclusion range mismatch")
    pilot_seeds = set(range(pilot_seed_start, pilot_seed_start + pilot_seed_count))
    if any(pilot_seeds.intersection(plan.master_seeds) for plan in plans_tuple):
        raise FinalProtocolError("Final seed schedules overlap the frozen pilot")

    matrix = _require_mapping(payload.get("matrix"), "matrix")
    engine_hands_per_unit = _require_int(matrix.get("engine_hands_per_paired_unit"), "matrix.engine_hands_per_paired_unit", positive=True)
    planned_units = _require_int(matrix.get("planned_independent_units"), "matrix.planned_independent_units", positive=True)
    planned_hands = _require_int(matrix.get("planned_simulated_engine_hands"), "matrix.planned_simulated_engine_hands", positive=True)
    if engine_hands_per_unit != 2:
        raise FinalProtocolError("Each paired_deal unit must contain exactly two engine hands")
    if matrix.get("cell_count") != 10 or matrix.get("opponent_profiles_per_hero") != 5:
        raise FinalProtocolError("Final matrix cell/profile counts are inconsistent")
    expected_units = len(opponent_profile_ids) * sum(plan.paired_units_per_profile for plan in plans_tuple)
    if planned_units != expected_units or planned_hands != expected_units * engine_hands_per_unit:
        raise FinalProtocolError("Final matrix totals are inconsistent with frozen allocations")

    failures = _require_mapping(payload.get("failure_handling"), "failure_handling")
    illegal_action_policy = _require_string(failures.get("illegal_action_policy"), "failure_handling.illegal_action_policy")
    replacement_seed_policy = _require_string(failures.get("replacement_seed_policy"), "failure_handling.replacement_seed_policy")
    if illegal_action_policy != IllegalActionPolicy.EXCLUDE_RUN.value or replacement_seed_policy != "forbidden":
        raise FinalProtocolError("Final failed units must be retained with no replacement seeds")
    if not _require_bool(
        failures.get("failed_units_retained_in_report"),
        "failure_handling.failed_units_retained_in_report",
    ):
        raise FinalProtocolError("Failed final units must remain visible in the report")

    reporting = _require_mapping(payload.get("reporting"), "reporting")
    if reporting.get("primary") != ["baseline_synthetic_population_mean_ev_bb100"]:
        raise FinalProtocolError("Primary reporting estimand mismatch")
    if reporting.get("secondary_descriptive") != [
        "balanced_population_mean_ev_bb100",
        "per_profile_ev_bb100",
        "population_median_ev_bb100",
        "r10_bb100",
        "q25_bb100",
        "population_sd_bb100",
        "empirical_minimum_bb100",
    ]:
        raise FinalProtocolError("Secondary descriptive reporting contract mismatch")
    if reporting.get("r10_is_descriptive_robustness_metric") is not True:
        raise FinalProtocolError("R10 must remain labelled as descriptive robustness")
    if reporting.get("pilot_results_excluded") is not True:
        raise FinalProtocolError("Pilot results must remain excluded from final reporting")

    scope = _require_mapping(payload.get("scope_limitations"), "scope_limitations")
    if _require_bool(scope.get("complete_dtest_population_benchmark"), "scope_limitations.complete_dtest_population_benchmark"):
        raise FinalProtocolError("Five static profiles must not be represented as the full D_test benchmark")
    if _require_bool(scope.get("meets_minimum_10_independent_opponent_samples"), "scope_limitations.meets_minimum_10_independent_opponent_samples"):
        raise FinalProtocolError("Five static profiles do not satisfy the >=10 D_test sample guidance")
    if not _require_bool(scope.get("tail_precision_warning"), "scope_limitations.tail_precision_warning"):
        raise FinalProtocolError("Five-profile R10 tail precision warning must remain enabled")
    if _require_bool(scope.get("full_section_11_2_benchmark_matrix"), "scope_limitations.full_section_11_2_benchmark_matrix"):
        raise FinalProtocolError("Phase-1 v1 is not the complete Section 11.2 benchmark matrix")
    if scope.get("static_profile_count") != 5:
        raise FinalProtocolError("Static-profile scope count must remain five")
    if any(
        scope.get(field) is not False
        for field in (
            "randomized_population_frozen",
            "stress_population_frozen",
            "holdout_population_frozen",
        )
    ):
        raise FinalProtocolError("Unfrozen later populations must not be claimed by Phase-1 v1")
    if scope.get("interpretation") != "scoped_phase1_static_profile_baseline_measurement_not_the_complete_population_benchmark":
        raise FinalProtocolError("Scope interpretation disclosure mismatch")

    if payload.get("population_evaluation_version") != POPULATION_EVALUATION_VERSION:
        raise FinalProtocolError("population_evaluation_version mismatch")

    return Phase1BaselineExperimentProtocol(
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        protocol_hash=stored_hash,
        status=status,
        purpose=purpose,
        statistical_unit=statistical_unit,
        game_config_hash=game_hash,
        primary_population_id=primary_id,
        primary_population_hash=primary_hash,
        secondary_population_ids=secondary_ids,
        opponent_profile_ids=opponent_profile_ids,
        agent_plans=plans_tuple,
        pilot_variance_cells=cells_tuple,
        pilot_protocol_hash=pilot_protocol_hash,
        pilot_report_hash=pilot_report_hash,
        gate6_report_hash=gate6_report_hash,
        confidence_level=confidence_level,
        target_half_width_bb100=target_half_width,
        illustrative_project_half_width_bb100=illustrative,
        planned_independent_units=planned_units,
        planned_simulated_engine_hands=planned_hands,
        engine_hands_per_paired_unit=engine_hands_per_unit,
        seed_schedule_version=PHASE1_FINAL_SEED_SCHEDULE_VERSION,
        replacement_seed_policy=replacement_seed_policy,
        illegal_action_policy=illegal_action_policy,
    )


def verify_pilot_report_binding(
    protocol: Phase1BaselineExperimentProtocol,
    pilot_report_path: str | Path,
) -> None:
    """Verify that final planning evidence is exactly the completed frozen pilot."""
    raw = json.loads(Path(pilot_report_path).read_text(encoding="utf-8"))
    envelope = _require_mapping(raw, "pilot report file")
    report = _require_mapping(envelope.get("report"), "pilot report")
    stored_hash = _require_sha256(envelope.get("report_hash"), "pilot report_hash")
    if sha256_canonical(report) != stored_hash or stored_hash != protocol.pilot_report_hash:
        raise FinalProtocolError("Pilot report hash does not match the frozen final planning basis")
    if report.get("report_version") != "phase1-pilot-execution-report-v1":
        raise FinalProtocolError("Pilot report version mismatch")
    if (
        report.get("protocol_hash") != protocol.pilot_protocol_hash
        or report.get("gate6_report_hash") != protocol.gate6_report_hash
        or report.get("complete") is not True
        or report.get("failure_free") is not True
        or report.get("completed_jobs") != 1000
        or report.get("valid_units") != 1000
        or report.get("failed_units") != 0
        or report.get("recorded_engine_hands") != 2000
    ):
        raise FinalProtocolError("Pilot report completion/provenance evidence mismatch")
    scientific_use = _require_mapping(report.get("scientific_use"), "pilot report scientific_use")
    if (
        scientific_use.get("contains_confirmatory_inference") is not False
        or scientific_use.get("contains_final_benchmark_estimate") is not False
        or scientific_use.get("final_sample_size_planning") is not True
        or scientific_use.get("variance_estimation") is not True
    ):
        raise FinalProtocolError("Pilot report scientific-use flags are incompatible")
    raw_cells = report.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != len(protocol.pilot_variance_cells):
        raise FinalProtocolError("Pilot report cell evidence count mismatch")
    actual = {
        (str(cell["hero_id"]), str(cell["opponent_profile_id"])): (
            float(cell["std_difference_bb100"]),
            int(cell["valid_units"]),
        )
        for cell in raw_cells
    }
    expected = {
        (cell.hero_id, cell.opponent_profile_id): (
            cell.std_difference_bb100,
            cell.valid_units,
        )
        for cell in protocol.pilot_variance_cells
    }
    if actual != expected:
        raise FinalProtocolError("Pilot report variance evidence differs from final planning basis")


def iter_final_jobs(protocol: Phase1BaselineExperimentProtocol) -> tuple[FinalJob, ...]:
    """Expand the frozen final allocation into canonical hero/profile/seed jobs."""
    jobs = tuple(
        FinalJob(
            cell_id=f"{plan.agent_id}-vs-{profile_id}",
            hero_id=plan.agent_id,
            hero_version=plan.agent_version,
            opponent_profile_id=profile_id,
            opponent_profile_version=STATIC_PROFILE_VERSION,
            master_seed=seed,
            paired_units=1,
            simulated_engine_hands=protocol.engine_hands_per_paired_unit,
        )
        for plan in protocol.agent_plans
        for profile_id in protocol.opponent_profile_ids
        for seed in plan.master_seeds
    )
    if len(jobs) != protocol.planned_independent_units:
        raise FinalProtocolError("Expanded final job count does not match the frozen protocol")
    return jobs
