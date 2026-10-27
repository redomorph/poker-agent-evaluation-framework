"""Frozen Phase-1 pilot protocol loading, validation, and job expansion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokerbench.agents.hand_strength_baseline import HAND_STRENGTH_BASELINE_VERSION
from pokerbench.agents.profiles import STATIC_PROFILE_VERSION
from pokerbench.agents.random_baseline import RANDOM_BASELINE_VERSION
from pokerbench.agents.strength_gates import PARAMETRIC_POLICY_VERSION
from pokerbench.engine.config import BASELINE_GAME_CONFIG, config_hash
from pokerbench.evaluation.population import (
    POPULATION_WEIGHT_SCALE,
    DiscretePopulationDefinition,
)
from pokerbench.runner import IllegalActionPolicy, StatisticalUnit


FROZEN_POPULATION_FILE_VERSION = "frozen-population-file-v1"
PHASE1_PILOT_PROTOCOL_FILE_VERSION = "phase1-pilot-protocol-file-v1"
PHASE1_PILOT_PROTOCOL_VERSION = "phase1-pilot-protocol-v1"
PHASE1_PILOT_SEED_SCHEDULE_VERSION = "phase1-pilot-seeds-v1"
PHASE1_PILOT_RNG_CONTRACT = "event-indexed-crn-v1"
PHASE1_PILOT_VARIANCE_ESTIMATOR = (
    "sample-sd-paired-difference-bb100-ddof1-v1"
)
PILOT_MIN_SIMULATED_HANDS = 1_000
PILOT_MAX_SIMULATED_HANDS = 5_000


class PilotProtocolError(ValueError):
    """Raised when frozen Phase-1 pilot evidence is malformed or inconsistent."""


@dataclass(frozen=True)
class PilotAgentReference:
    """One frozen static hero agent participating in the pilot."""

    agent_id: str
    agent_version: str
    adaptive: bool


@dataclass(frozen=True)
class PilotPopulationReference:
    """Hash-bound reference to one frozen opponent population."""

    population_id: str
    version: str
    population_hash: str


@dataclass(frozen=True)
class Phase1PilotProtocol:
    """Validated immutable Phase-1 pilot design."""

    protocol_id: str
    protocol_version: str
    status: str
    purpose: str
    protocol_hash: str

    final_experiment_data: bool
    eligible_for_confirmatory_analysis: bool

    game_abstraction_version: str
    game_config_hash: str
    statistical_unit: str

    hero_agents: tuple[PilotAgentReference, ...]
    opponent_profile_ids: tuple[str, ...]
    static_profile_version: str
    parametric_policy_version: str
    gate6_pass_required: bool
    gate6_report_hash_binding: str

    primary_population: PilotPopulationReference
    secondary_populations: tuple[PilotPopulationReference, ...]

    rng_contract: str
    crn_mode: str
    deal_namespace: str
    bot_a_namespace: str
    bot_b_namespace: str
    agent_event_key: str
    seed_schedule_version: str
    master_seeds: tuple[int, ...]
    reuse_same_seed_schedule_across_cells: bool

    paired_units_per_cell: int
    engine_hands_per_paired_unit: int
    cell_count: int
    planned_independent_units: int
    planned_simulated_engine_hands: int

    illegal_action_policy: str
    replacement_seed_policy: str

    variance_estimator: str
    runtime_measurement: bool
    allowed_scientific_use: tuple[str, ...]
    forbidden_scientific_use: tuple[str, ...]

    minimum_relevant_difference_bb100: float | None
    precision_target_half_width_bb100: float | None
    final_independent_units: int | None

    required_run_provenance: tuple[str, ...]


@dataclass(frozen=True)
class PilotJob:
    """One deterministic cell/seed job implied by the frozen pilot protocol."""

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


def canonical_file_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON file bytes with one trailing LF."""
    return (_canonical_json(value) + "\n").encode("utf-8")


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotProtocolError(f"{field_name} must be a JSON object")
    return value


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotProtocolError(f"{field_name} must be a non-empty string")
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PilotProtocolError(f"{field_name} must be boolean")
    return value


def _require_int(value: Any, field_name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotProtocolError(f"{field_name} must be an integer")
    if positive and value <= 0:
        raise PilotProtocolError(f"{field_name} must be positive")
    return value


def _require_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PilotProtocolError(f"{field_name} must be a non-empty JSON array")
    values = tuple(_require_string(item, field_name) for item in value)
    if len(set(values)) != len(values):
        raise PilotProtocolError(f"{field_name} must contain unique values")
    return values


def _require_sha256(value: Any, field_name: str) -> str:
    text = _require_string(value, field_name)
    if len(text) != 64:
        raise PilotProtocolError(f"{field_name} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise PilotProtocolError(f"{field_name} must be hexadecimal") from exc
    return text.lower()


def load_frozen_population(path: str | Path) -> DiscretePopulationDefinition:
    """Load and integrity-check one machine-readable frozen population."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    envelope = _require_mapping(raw, "population file")

    if envelope.get("file_version") != FROZEN_POPULATION_FILE_VERSION:
        raise PilotProtocolError("Unsupported frozen population file_version")

    stored_hash = _require_sha256(envelope.get("population_hash"), "population_hash")
    payload = _require_mapping(envelope.get("population"), "population")
    actual_hash = _sha256_canonical(payload)
    if stored_hash != actual_hash:
        raise PilotProtocolError("population_hash does not match population payload")

    if payload.get("weight_scale") != POPULATION_WEIGHT_SCALE:
        raise PilotProtocolError("population weight_scale is inconsistent")

    population_id = _require_string(payload.get("population_id"), "population_id")
    version = _require_string(payload.get("version"), "population version")
    raw_weights = payload.get("weights_ppm")
    if not isinstance(raw_weights, list) or not raw_weights:
        raise PilotProtocolError("weights_ppm must be a non-empty JSON array")

    weights: list[tuple[str, int]] = []
    for index, raw_weight in enumerate(raw_weights):
        entry = _require_mapping(raw_weight, f"weights_ppm[{index}]")
        profile_id = _require_string(
            entry.get("profile_id"), f"weights_ppm[{index}].profile_id"
        )
        weight = _require_int(
            entry.get("weight_ppm"), f"weights_ppm[{index}].weight_ppm", positive=True
        )
        weights.append((profile_id, weight))

    population = DiscretePopulationDefinition(
        population_id=population_id,
        version=version,
        weights_ppm=tuple(weights),
    )
    if population.population_hash != stored_hash:
        raise PilotProtocolError(
            "population_hash is not compatible with population-evaluation-v1"
        )
    return population


def _population_reference(
    raw: Any,
    *,
    field_name: str,
    populations: Mapping[str, DiscretePopulationDefinition],
) -> PilotPopulationReference:
    value = _require_mapping(raw, field_name)
    population_id = _require_string(value.get("population_id"), f"{field_name}.population_id")
    version = _require_string(value.get("version"), f"{field_name}.version")
    population_hash = _require_sha256(
        value.get("population_hash"), f"{field_name}.population_hash"
    )

    population = populations.get(population_id)
    if population is None:
        raise PilotProtocolError(f"{field_name} has no supplied frozen population")
    if population.version != version or population.population_hash != population_hash:
        raise PilotProtocolError(f"{field_name} does not match supplied frozen population")

    return PilotPopulationReference(
        population_id=population_id,
        version=version,
        population_hash=population_hash,
    )


def load_phase1_pilot_protocol(
    path: str | Path,
    *,
    populations: Mapping[str, DiscretePopulationDefinition],
) -> Phase1PilotProtocol:
    """Load and validate the frozen Phase-1 variance-estimation pilot."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    envelope = _require_mapping(raw, "pilot protocol file")

    if envelope.get("file_version") != PHASE1_PILOT_PROTOCOL_FILE_VERSION:
        raise PilotProtocolError("Unsupported pilot protocol file_version")

    stored_hash = _require_sha256(envelope.get("protocol_hash"), "protocol_hash")
    payload = _require_mapping(envelope.get("protocol"), "protocol")
    actual_hash = _sha256_canonical(payload)
    if stored_hash != actual_hash:
        raise PilotProtocolError("protocol_hash does not match protocol payload")

    protocol_id = _require_string(payload.get("protocol_id"), "protocol_id")
    protocol_version = _require_string(payload.get("protocol_version"), "protocol_version")
    if protocol_version != PHASE1_PILOT_PROTOCOL_VERSION:
        raise PilotProtocolError("Unsupported protocol_version")
    status = _require_string(payload.get("status"), "status")
    purpose = _require_string(payload.get("purpose"), "purpose")
    if status != "frozen" or purpose != "variance_and_runtime_estimation_only":
        raise PilotProtocolError("Phase-1 pilot must be frozen and variance-only")

    scientific_use = _require_mapping(payload.get("scientific_use"), "scientific_use")
    final_experiment_data = _require_bool(
        scientific_use.get("final_experiment_data"), "scientific_use.final_experiment_data"
    )
    eligible_for_confirmatory_analysis = _require_bool(
        scientific_use.get("eligible_for_confirmatory_analysis"),
        "scientific_use.eligible_for_confirmatory_analysis",
    )
    if final_experiment_data or eligible_for_confirmatory_analysis:
        raise PilotProtocolError("Pilot evidence must not be final or confirmatory data")

    game = _require_mapping(payload.get("game"), "game")
    game_abstraction_version = _require_string(
        game.get("abstraction_version"), "game.abstraction_version"
    )
    game_config_hash = _require_sha256(game.get("game_config_hash"), "game.game_config_hash")
    if game_abstraction_version != BASELINE_GAME_CONFIG.abstraction_version:
        raise PilotProtocolError("Pilot game abstraction does not match BASELINE_GAME_CONFIG")
    if game_config_hash != config_hash(BASELINE_GAME_CONFIG):
        raise PilotProtocolError("Pilot game_config_hash does not match BASELINE_GAME_CONFIG")

    statistical_unit = _require_string(payload.get("statistical_unit"), "statistical_unit")
    if statistical_unit != StatisticalUnit.PAIRED_DEAL.value:
        raise PilotProtocolError("Phase-1 static pilot requires paired_deal statistical units")

    raw_heroes = payload.get("hero_agents")
    if not isinstance(raw_heroes, list) or not raw_heroes:
        raise PilotProtocolError("hero_agents must be a non-empty JSON array")
    hero_agents: list[PilotAgentReference] = []
    for index, raw_hero in enumerate(raw_heroes):
        hero = _require_mapping(raw_hero, f"hero_agents[{index}]")
        hero_agents.append(
            PilotAgentReference(
                agent_id=_require_string(hero.get("agent_id"), f"hero_agents[{index}].agent_id"),
                agent_version=_require_string(
                    hero.get("agent_version"), f"hero_agents[{index}].agent_version"
                ),
                adaptive=_require_bool(hero.get("adaptive"), f"hero_agents[{index}].adaptive"),
            )
        )
    if tuple((item.agent_id, item.agent_version, item.adaptive) for item in hero_agents) != (
        ("random", RANDOM_BASELINE_VERSION, False),
        ("hand_strength", HAND_STRENGTH_BASELINE_VERSION, False),
    ):
        raise PilotProtocolError("hero_agents do not match the frozen Phase-1 baselines")

    opponents = _require_mapping(payload.get("opponents"), "opponents")
    opponent_profile_ids = _require_string_tuple(
        opponents.get("profile_ids"), "opponents.profile_ids"
    )
    if opponent_profile_ids != ("nit", "tag", "lag", "calling_station", "maniac"):
        raise PilotProtocolError("opponents.profile_ids must match the five frozen static profiles")
    static_profile_version = _require_string(
        opponents.get("static_profile_version"), "opponents.static_profile_version"
    )
    parametric_policy_version = _require_string(
        opponents.get("parametric_policy_version"), "opponents.parametric_policy_version"
    )
    if static_profile_version != STATIC_PROFILE_VERSION:
        raise PilotProtocolError("static profile version mismatch")
    if parametric_policy_version != PARAMETRIC_POLICY_VERSION:
        raise PilotProtocolError("parametric policy version mismatch")
    gate6_pass_required = _require_bool(
        opponents.get("gate6_pass_required"), "opponents.gate6_pass_required"
    )
    gate6_report_hash_binding = _require_string(
        opponents.get("gate6_report_hash_binding"), "opponents.gate6_report_hash_binding"
    )
    if not gate6_pass_required or gate6_report_hash_binding != "bind_before_first_pilot_hand":
        raise PilotProtocolError("Gate-6 PASS evidence must be bound before pilot execution")

    population_block = _require_mapping(payload.get("populations"), "populations")
    primary_population = _population_reference(
        population_block.get("primary"),
        field_name="populations.primary",
        populations=populations,
    )
    raw_secondary = population_block.get("secondary_descriptive")
    if not isinstance(raw_secondary, list):
        raise PilotProtocolError("populations.secondary_descriptive must be a JSON array")
    secondary_populations = tuple(
        _population_reference(
            item,
            field_name=f"populations.secondary_descriptive[{index}]",
            populations=populations,
        )
        for index, item in enumerate(raw_secondary)
    )

    rng = _require_mapping(payload.get("rng"), "rng")
    rng_contract = _require_string(rng.get("rng_contract"), "rng.rng_contract")
    crn_mode = _require_string(rng.get("crn_mode"), "rng.crn_mode")
    deal_namespace = _require_string(rng.get("deal_namespace"), "rng.deal_namespace")
    bot_a_namespace = _require_string(rng.get("bot_a_namespace"), "rng.bot_a_namespace")
    bot_b_namespace = _require_string(rng.get("bot_b_namespace"), "rng.bot_b_namespace")
    agent_event_key = _require_string(rng.get("agent_event_key"), "rng.agent_event_key")
    seed_schedule_version = _require_string(
        rng.get("seed_schedule_version"), "rng.seed_schedule_version"
    )
    raw_seeds = rng.get("master_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise PilotProtocolError("rng.master_seeds must be a non-empty JSON array")
    master_seeds = tuple(
        _require_int(seed, f"rng.master_seeds[{index}]")
        for index, seed in enumerate(raw_seeds)
    )
    if len(set(master_seeds)) != len(master_seeds):
        raise PilotProtocolError("rng.master_seeds must be unique")
    reuse_seed_schedule = _require_bool(
        rng.get("reuse_same_seed_schedule_across_cells"),
        "rng.reuse_same_seed_schedule_across_cells",
    )
    if (
        rng_contract != PHASE1_PILOT_RNG_CONTRACT
        or crn_mode != "identical_realisations"
        or deal_namespace != "DEAL_SCHEDULE"
        or bot_a_namespace != "RNG_BOT_A"
        or bot_b_namespace != "RNG_BOT_B"
        or agent_event_key != "(bot_id,hand_id,decision_index,purpose)"
        or seed_schedule_version != PHASE1_PILOT_SEED_SCHEDULE_VERSION
        or not reuse_seed_schedule
    ):
        raise PilotProtocolError("Pilot RNG/CRN contract does not match the frozen design")

    matrix = _require_mapping(payload.get("matrix"), "matrix")
    paired_units_per_cell = _require_int(
        matrix.get("paired_units_per_cell"), "matrix.paired_units_per_cell", positive=True
    )
    engine_hands_per_paired_unit = _require_int(
        matrix.get("engine_hands_per_paired_unit"),
        "matrix.engine_hands_per_paired_unit",
        positive=True,
    )
    cell_count = _require_int(matrix.get("cell_count"), "matrix.cell_count", positive=True)
    planned_independent_units = _require_int(
        matrix.get("planned_independent_units"), "matrix.planned_independent_units", positive=True
    )
    planned_simulated_engine_hands = _require_int(
        matrix.get("planned_simulated_engine_hands"),
        "matrix.planned_simulated_engine_hands",
        positive=True,
    )
    expected_cells = len(hero_agents) * len(opponent_profile_ids)
    if cell_count != expected_cells:
        raise PilotProtocolError("matrix.cell_count is inconsistent")
    if len(master_seeds) != paired_units_per_cell:
        raise PilotProtocolError("master seed count must equal paired_units_per_cell")
    if planned_independent_units != cell_count * paired_units_per_cell:
        raise PilotProtocolError("matrix.planned_independent_units is inconsistent")
    if engine_hands_per_paired_unit != 2:
        raise PilotProtocolError("paired_deal pilot units must represent exactly two engine hands")
    if planned_simulated_engine_hands != planned_independent_units * engine_hands_per_paired_unit:
        raise PilotProtocolError("matrix.planned_simulated_engine_hands is inconsistent")
    if not PILOT_MIN_SIMULATED_HANDS <= planned_simulated_engine_hands <= PILOT_MAX_SIMULATED_HANDS:
        raise PilotProtocolError("pilot simulated hands must remain inside the 1,000-5,000 guidance")

    failures = _require_mapping(payload.get("failure_handling"), "failure_handling")
    illegal_action_policy = _require_string(
        failures.get("illegal_action_policy"), "failure_handling.illegal_action_policy"
    )
    replacement_seed_policy = _require_string(
        failures.get("replacement_seed_policy"), "failure_handling.replacement_seed_policy"
    )
    if illegal_action_policy != IllegalActionPolicy.EXCLUDE_RUN.value:
        raise PilotProtocolError("Pilot illegal-action policy must remain exclude_run")
    if replacement_seed_policy != "forbidden":
        raise PilotProtocolError("Pilot replacement seeds are forbidden")

    analysis = _require_mapping(payload.get("pilot_analysis"), "pilot_analysis")
    variance_estimator = _require_string(
        analysis.get("variance_estimator"), "pilot_analysis.variance_estimator"
    )
    runtime_measurement = _require_bool(
        analysis.get("runtime_measurement"), "pilot_analysis.runtime_measurement"
    )
    allowed_scientific_use = _require_string_tuple(
        analysis.get("allowed_scientific_use"), "pilot_analysis.allowed_scientific_use"
    )
    forbidden_scientific_use = _require_string_tuple(
        analysis.get("forbidden_scientific_use"), "pilot_analysis.forbidden_scientific_use"
    )
    if variance_estimator != PHASE1_PILOT_VARIANCE_ESTIMATOR or not runtime_measurement:
        raise PilotProtocolError("Pilot analysis contract mismatch")

    final_design = _require_mapping(payload.get("final_design"), "final_design")
    minimum_relevant_difference_bb100 = final_design.get("minimum_relevant_difference_bb100")
    precision_target_half_width_bb100 = final_design.get("precision_target_half_width_bb100")
    final_independent_units = final_design.get("final_independent_units")
    if any(
        value is not None
        for value in (
            minimum_relevant_difference_bb100,
            precision_target_half_width_bb100,
            final_independent_units,
        )
    ):
        raise PilotProtocolError(
            "Pilot final design fields must remain null until pilot variance is available"
        )

    required_run_provenance = _require_string_tuple(
        payload.get("required_run_provenance"), "required_run_provenance"
    )

    return Phase1PilotProtocol(
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        status=status,
        purpose=purpose,
        protocol_hash=stored_hash,
        final_experiment_data=final_experiment_data,
        eligible_for_confirmatory_analysis=eligible_for_confirmatory_analysis,
        game_abstraction_version=game_abstraction_version,
        game_config_hash=game_config_hash,
        statistical_unit=statistical_unit,
        hero_agents=tuple(hero_agents),
        opponent_profile_ids=opponent_profile_ids,
        static_profile_version=static_profile_version,
        parametric_policy_version=parametric_policy_version,
        gate6_pass_required=gate6_pass_required,
        gate6_report_hash_binding=gate6_report_hash_binding,
        primary_population=primary_population,
        secondary_populations=secondary_populations,
        rng_contract=rng_contract,
        crn_mode=crn_mode,
        deal_namespace=deal_namespace,
        bot_a_namespace=bot_a_namespace,
        bot_b_namespace=bot_b_namespace,
        agent_event_key=agent_event_key,
        seed_schedule_version=seed_schedule_version,
        master_seeds=master_seeds,
        reuse_same_seed_schedule_across_cells=reuse_seed_schedule,
        paired_units_per_cell=paired_units_per_cell,
        engine_hands_per_paired_unit=engine_hands_per_paired_unit,
        cell_count=cell_count,
        planned_independent_units=planned_independent_units,
        planned_simulated_engine_hands=planned_simulated_engine_hands,
        illegal_action_policy=illegal_action_policy,
        replacement_seed_policy=replacement_seed_policy,
        variance_estimator=variance_estimator,
        runtime_measurement=runtime_measurement,
        allowed_scientific_use=allowed_scientific_use,
        forbidden_scientific_use=forbidden_scientific_use,
        minimum_relevant_difference_bb100=None,
        precision_target_half_width_bb100=None,
        final_independent_units=None,
        required_run_provenance=required_run_provenance,
    )


def iter_pilot_jobs(protocol: Phase1PilotProtocol) -> tuple[PilotJob, ...]:
    """Expand the frozen matrix into canonical hero/profile/seed execution jobs."""
    if not isinstance(protocol, Phase1PilotProtocol):
        raise TypeError("protocol must be a Phase1PilotProtocol")

    jobs = tuple(
        PilotJob(
            cell_id=f"{hero.agent_id}-vs-{profile_id}",
            hero_id=hero.agent_id,
            hero_version=hero.agent_version,
            opponent_profile_id=profile_id,
            opponent_profile_version=protocol.static_profile_version,
            master_seed=seed,
            paired_units=1,
            simulated_engine_hands=protocol.engine_hands_per_paired_unit,
        )
        for hero in protocol.hero_agents
        for profile_id in protocol.opponent_profile_ids
        for seed in protocol.master_seeds
    )
    if len(jobs) != protocol.planned_independent_units:
        raise PilotProtocolError("Expanded pilot job count does not match protocol")
    return jobs
