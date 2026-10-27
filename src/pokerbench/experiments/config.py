# src/pokerbench/experiments/config.py
"""
Frozen experiment configuration and deterministic persistence.

A scientific benchmark is represented by one complete immutable experiment
definition. The configuration includes:

- game configuration and bet abstraction;
- bot and opponent versions/configurations;
- opponent population definition;
- deterministic master seeds;
- RNG configuration;
- explicit statistical unit;
- sample-size plan;
- power and precision targets;
- statistical evaluation method;
- calibration record;
- illegal-action policy;
- software, dependency, and hardware metadata.

The complete configuration is serialized to canonical normalized JSON and
hashed using SHA-256.

Persistence files contain both:

    configuration
    configuration_hash

The hash is verified when a persisted experiment configuration is loaded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import string
from collections.abc import Mapping
from dataclasses import (
    dataclass,
    fields,
    is_dataclass,
)
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

from pokerbench.engine.config import (
    Action,
    GameConfig,
    GameType,
)
from pokerbench.evaluation.statistics import (
    BOOTSTRAP_PERCENTILE_METHOD,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CONFIDENCE_LEVEL,
    EVALUATION_VERSION,
)
from pokerbench.runner import (
    IllegalActionPolicy,
    StatisticalUnit,
)


EXPERIMENT_CONFIG_SCHEMA_VERSION = "experiment-config-v1"
EXPERIMENT_CONFIG_FILE_VERSION = "experiment-config-file-v1"


JSONValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | list["JSONValue"]
    | dict[str, "JSONValue"]
)


class CRNMode(StrEnum):
    """
    Relationship between paired experimental conditions.

    IDENTICAL_REALISATIONS
        Corresponding conditions reuse the same pre-defined chance
        realisations.

    SEED_SYNCHRONISED
        Random streams are synchronised by seed but identical chance
        realisations cannot be guaranteed.

    INDEPENDENT
        Conditions use independent chance realisations.
    """

    IDENTICAL_REALISATIONS = "identical_realisations"
    SEED_SYNCHRONISED = "seed_synchronised"
    INDEPENDENT = "independent"


def _require_non_empty(
    value: str,
    field_name: str,
) -> None:
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string"
        )

    if not value.strip():
        raise ValueError(
            f"{field_name} must not be empty"
        )


def _require_positive_int(
    value: int,
    field_name: str,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{field_name} must be an integer"
        )

    if value <= 0:
        raise ValueError(
            f"{field_name} must be positive"
        )


def _require_non_negative_int(
    value: int,
    field_name: str,
) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise TypeError(
            f"{field_name} must be an integer"
        )

    if value < 0:
        raise ValueError(
            f"{field_name} must be non-negative"
        )


def _require_probability(
    value: float,
    field_name: str,
) -> None:
    if not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{field_name} must be numeric"
        )

    if not (
        0.0
        < float(value)
        < 1.0
    ):
        raise ValueError(
            f"{field_name} must satisfy 0 < value < 1"
        )


def _validate_sha256(
    value: str,
    field_name: str,
) -> None:
    _require_non_empty(
        value,
        field_name,
    )

    if len(value) != 64:
        raise ValueError(
            f"{field_name} must contain a 64-character SHA-256 digest"
        )

    valid = set(
        string.hexdigits
    )

    if any(
        character not in valid
        for character in value
    ):
        raise ValueError(
            f"{field_name} must be hexadecimal"
        )


def _normalize_json_value(
    value: Any,
) -> JSONValue:
    """
    Validate and normalize a generic JSON-compatible value.
    """
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            raise ValueError(
                "JSON floating-point values must be finite"
            )

        return value

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        list,
    ):
        return [
            _normalize_json_value(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            _normalize_json_value(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        Mapping,
    ):
        normalized: dict[
            str,
            JSONValue,
        ] = {}

        for key, item in value.items():
            if not isinstance(
                key,
                str,
            ):
                raise TypeError(
                    "JSON object keys must be strings"
                )

            normalized[key] = (
                _normalize_json_value(
                    item
                )
            )

        return normalized

    raise TypeError(
        f"Unsupported JSON value type: "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class CanonicalJSON:
    """
    Immutable, canonically serialized arbitrary JSON value.

    Semantically equivalent JSON values therefore compare identically
    regardless of whitespace or dictionary insertion order.
    """

    text: str

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.text,
            str,
        ):
            raise TypeError(
                "CanonicalJSON.text must be a string"
            )

        try:
            raw = json.loads(
                self.text
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "CanonicalJSON.text must contain valid JSON"
            ) from exc

        normalized = (
            _normalize_json_value(
                raw
            )
        )

        canonical = json.dumps(
            normalized,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )

        object.__setattr__(
            self,
            "text",
            canonical,
        )

    @classmethod
    def from_value(
        cls,
        value: JSONValue,
    ) -> "CanonicalJSON":
        normalized = (
            _normalize_json_value(
                value
            )
        )

        text = json.dumps(
            normalized,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )

        return cls(text)

    def to_value(
        self,
    ) -> JSONValue:
        return cast(
            JSONValue,
            json.loads(
                self.text
            ),
        )


@dataclass(frozen=True)
class AgentExperimentConfig:
    """
    Frozen description of one agent participating in an experiment.

    The agent configuration contains its behavioural parameters and its
    declared execution budgets.

    Budget enforcement itself belongs to the runner/agent execution layer.
    This object only makes the frozen scientific configuration explicit.
    """

    name: str
    version: str

    parameters: CanonicalJSON

    simulation_budget: int | None
    time_budget_ms: int | None
    timeout_policy: str

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.name,
            "name",
        )
        _require_non_empty(
            self.version,
            "version",
        )
        _require_non_empty(
            self.timeout_policy,
            "timeout_policy",
        )

        if self.simulation_budget is not None:
            _require_non_negative_int(
                self.simulation_budget,
                "simulation_budget",
            )

        if self.time_budget_ms is not None:
            _require_positive_int(
                self.time_budget_ms,
                "time_budget_ms",
            )


@dataclass(frozen=True)
class PopulationConfig:
    """
    Frozen opponent-population definition.
    """

    population_id: str
    version: str
    distribution: CanonicalJSON
    population_seed: int | None

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.population_id,
            "population_id",
        )
        _require_non_empty(
            self.version,
            "population version",
        )

        if self.population_seed is not None:
            if (
                not isinstance(
                    self.population_seed,
                    int,
                )
                or isinstance(
                    self.population_seed,
                    bool,
                )
            ):
                raise TypeError(
                    "population_seed must be an integer or None"
                )


@dataclass(frozen=True)
class RNGConfiguration:
    """
    Frozen RNG/CRN definition.
    """

    version: str
    crn_mode: CRNMode

    deal_namespace: str = "DEAL_SCHEDULE"
    bot_a_namespace: str = "RNG_BOT_A"
    bot_b_namespace: str = "RNG_BOT_B"

    agent_event_key: str = (
        "(bot_id,hand_id,decision_index,purpose)"
    )

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.version,
            "RNG version",
        )
        _require_non_empty(
            self.deal_namespace,
            "deal_namespace",
        )
        _require_non_empty(
            self.bot_a_namespace,
            "bot_a_namespace",
        )
        _require_non_empty(
            self.bot_b_namespace,
            "bot_b_namespace",
        )
        _require_non_empty(
            self.agent_event_key,
            "agent_event_key",
        )


@dataclass(frozen=True)
class SampleSizePlan:
    """
    Pre-defined size of the complete experiment.

    The number of simulated engine hands and the number of independent
    statistical units are persisted separately.
    """

    planned_simulated_hands: int
    planned_independent_units: int

    def __post_init__(
        self,
    ) -> None:
        _require_positive_int(
            self.planned_simulated_hands,
            "planned_simulated_hands",
        )

        _require_positive_int(
            self.planned_independent_units,
            "planned_independent_units",
        )


@dataclass(frozen=True)
class PowerTarget:
    """
    Frozen power-analysis assumptions for a confirmatory comparison.

    This object records the inputs required by Section 3.5.

    It is optional at experiment level because power analysis is required
    only where a concrete hypothesis test is planned.
    """

    alpha: float
    beta: float

    minimum_relevant_difference_bb100: float
    estimated_sd_difference_bb100: float

    variance_source: str

    def __post_init__(
        self,
    ) -> None:
        _require_probability(
            self.alpha,
            "alpha",
        )
        _require_probability(
            self.beta,
            "beta",
        )

        if (
            self.minimum_relevant_difference_bb100
            <= 0.0
        ):
            raise ValueError(
                "minimum_relevant_difference_bb100 "
                "must be positive"
            )

        if (
            self.estimated_sd_difference_bb100
            <= 0.0
        ):
            raise ValueError(
                "estimated_sd_difference_bb100 "
                "must be positive"
            )

        _require_non_empty(
            self.variance_source,
            "variance_source",
        )


@dataclass(frozen=True)
class PrecisionTarget:
    """
    Frozen estimator-specific precision target.

    The specification explicitly requires persistence of:

    - estimand;
    - target CI half-width;
    - confidence level;
    - statistical unit;
    - variance estimator.
    """

    estimand: str
    target_ci_half_width_bb100: float
    confidence_level: float
    statistical_unit: StatisticalUnit
    variance_estimator: str

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.estimand,
            "estimand",
        )

        if (
            self.target_ci_half_width_bb100
            <= 0.0
        ):
            raise ValueError(
                "target_ci_half_width_bb100 "
                "must be positive"
            )

        _require_probability(
            self.confidence_level,
            "confidence_level",
        )

        _require_non_empty(
            self.variance_estimator,
            "variance_estimator",
        )


@dataclass(frozen=True)
class StatisticalMethodConfig:
    """
    Frozen statistical evaluation procedure.
    """

    bootstrap_seed: int

    method_version: str = (
        EVALUATION_VERSION
    )

    confidence_level: float = (
        DEFAULT_CONFIDENCE_LEVEL
    )

    bootstrap_resamples: int = (
        DEFAULT_BOOTSTRAP_RESAMPLES
    )

    bootstrap_percentile_method: str = (
        BOOTSTRAP_PERCENTILE_METHOD
    )

    effect_size: str = "d_z"

    def __post_init__(
        self,
    ) -> None:
        if (
            not isinstance(
                self.bootstrap_seed,
                int,
            )
            or isinstance(
                self.bootstrap_seed,
                bool,
            )
        ):
            raise TypeError(
                "bootstrap_seed must be an integer"
            )

        _require_non_empty(
            self.method_version,
            "method_version",
        )

        _require_probability(
            self.confidence_level,
            "confidence_level",
        )

        _require_positive_int(
            self.bootstrap_resamples,
            "bootstrap_resamples",
        )

        _require_non_empty(
            self.bootstrap_percentile_method,
            "bootstrap_percentile_method",
        )

        _require_non_empty(
            self.effect_size,
            "effect_size",
        )


@dataclass(frozen=True)
class CalibrationRecord:
    """
    Frozen calibration result attached to the experiment.

    For policies where calibration is not applicable, the result should
    explicitly say so rather than silently omitting the field.
    """

    version: str
    status: str
    results: CanonicalJSON

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.version,
            "calibration version",
        )
        _require_non_empty(
            self.status,
            "calibration status",
        )


@dataclass(frozen=True)
class ExecutionEnvironment:
    """
    Frozen software/dependency/hardware environment metadata.
    """

    python_version: str
    engine_version: str

    dependency_lockfile: str
    poetry_lock_hash: str

    git_commit: str

    cpu_model: str
    cpu_core_count: int
    cpu_architecture: str
    ram_bytes: int

    os_identifier: str

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.python_version,
            "python_version",
        )
        _require_non_empty(
            self.engine_version,
            "engine_version",
        )
        _require_non_empty(
            self.dependency_lockfile,
            "dependency_lockfile",
        )

        _validate_sha256(
            self.poetry_lock_hash,
            "poetry_lock_hash",
        )

        _require_non_empty(
            self.git_commit,
            "git_commit",
        )
        _require_non_empty(
            self.cpu_model,
            "cpu_model",
        )

        _require_positive_int(
            self.cpu_core_count,
            "cpu_core_count",
        )

        _require_non_empty(
            self.cpu_architecture,
            "cpu_architecture",
        )

        _require_positive_int(
            self.ram_bytes,
            "ram_bytes",
        )

        _require_non_empty(
            self.os_identifier,
            "os_identifier",
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Complete frozen scientific experiment definition.
    """

    experiment_id: str

    game_config: GameConfig

    bot: AgentExperimentConfig
    opponent: AgentExperimentConfig

    population: PopulationConfig

    master_seeds: tuple[int, ...]

    rng: RNGConfiguration

    adaptive: bool
    statistical_unit: StatisticalUnit

    sample_size: SampleSizePlan

    precision_target: PrecisionTarget
    power_target: PowerTarget | None

    statistical_method: StatisticalMethodConfig

    calibration: CalibrationRecord

    illegal_action_policy: IllegalActionPolicy

    environment: ExecutionEnvironment

    schema_version: str = (
        EXPERIMENT_CONFIG_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_non_empty(
            self.experiment_id,
            "experiment_id",
        )

        if (
            self.schema_version
            != EXPERIMENT_CONFIG_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported experiment configuration "
                f"schema version: {self.schema_version}"
            )

        if not self.master_seeds:
            raise ValueError(
                "master_seeds must contain at least one seed"
            )

        for seed in self.master_seeds:
            if (
                not isinstance(
                    seed,
                    int,
                )
                or isinstance(
                    seed,
                    bool,
                )
            ):
                raise TypeError(
                    "Every master seed must be an integer"
                )

        if (
            len(set(self.master_seeds))
            != len(self.master_seeds)
        ):
            raise ValueError(
                "master_seeds must be unique"
            )

        if (
            self.precision_target.statistical_unit
            != self.statistical_unit
        ):
            raise ValueError(
                "Precision target statistical unit must match "
                "the experiment statistical unit"
            )

        if not math.isclose(
            self.precision_target.confidence_level,
            self.statistical_method.confidence_level,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Precision-target confidence level must match "
                "the statistical-method confidence level"
            )

        if (
            self.adaptive
            and self.statistical_unit
            == StatisticalUnit.PAIRED_DEAL
        ):
            raise ValueError(
                "Adaptive/stateful experiments cannot use "
                "PAIRED_DEAL as their independent statistical unit"
            )

    @property
    def configuration_hash(
        self,
    ) -> str:
        return experiment_config_hash(
            self
        )


def _normalize_experiment_value(
    value: Any,
) -> JSONValue:
    """
    Convert experiment configuration values into canonical JSON values.
    """
    if isinstance(
        value,
        CanonicalJSON,
    ):
        return value.to_value()

    if isinstance(
        value,
        Decimal,
    ):
        return str(value)

    if isinstance(
        value,
        Enum,
    ):
        return cast(
            str,
            value.value,
        )

    if is_dataclass(value):
        normalized: dict[
            str,
            JSONValue,
        ] = {}

        for field_definition in fields(
            value
        ):
            normalized[
                field_definition.name
            ] = _normalize_experiment_value(
                getattr(
                    value,
                    field_definition.name,
                )
            )

        return normalized

    if isinstance(
        value,
        tuple,
    ):
        return [
            _normalize_experiment_value(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            _normalize_experiment_value(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): (
                _normalize_experiment_value(
                    item
                )
            )
            for key, item in value.items()
        }

    return _normalize_json_value(
        value
    )


def canonical_experiment_config_dict(
    config: ExperimentConfig,
) -> dict[str, JSONValue]:
    """
    Return the normalized experiment configuration as a dictionary.
    """
    normalized = (
        _normalize_experiment_value(
            config
        )
    )

    if not isinstance(
        normalized,
        dict,
    ):
        raise TypeError(
            "Experiment configuration normalization "
            "did not produce an object"
        )

    return normalized


def canonical_experiment_config_json(
    config: ExperimentConfig,
) -> str:
    """
    Return canonical sorted normalized JSON.
    """
    return json.dumps(
        canonical_experiment_config_dict(
            config
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    )


def experiment_config_hash(
    config: ExperimentConfig,
) -> str:
    """
    Return SHA-256 of the canonical frozen configuration.
    """
    canonical_json = (
        canonical_experiment_config_json(
            config
        )
    )

    return hashlib.sha256(
        canonical_json.encode(
            "utf-8"
        )
    ).hexdigest()


def sha256_file(
    path: str | Path,
) -> str:
    """
    Return the SHA-256 digest of a file.
    """
    file_path = Path(
        path
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            file_path
        )

    digest = hashlib.sha256()

    with file_path.open(
        "rb"
    ) as file_handle:
        while True:
            block = file_handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def capture_execution_environment(
    *,
    engine_version: str,
    git_commit: str,
    poetry_lock_path: str | Path,
    ram_bytes: int,
) -> ExecutionEnvironment:
    """
    Capture the deterministic execution-environment metadata available
    through the Python standard library.

    RAM is supplied explicitly so the framework never silently guesses a
    platform-specific memory value.

    Git commit and engine version are also explicit inputs because an
    experiment must record the exact intended scientific software version.
    """
    _require_non_empty(
        engine_version,
        "engine_version",
    )
    _require_non_empty(
        git_commit,
        "git_commit",
    )
    _require_positive_int(
        ram_bytes,
        "ram_bytes",
    )

    poetry_lock = Path(
        poetry_lock_path
    )

    cpu_model = (
        platform.processor().strip()
        or os.environ.get(
            "PROCESSOR_IDENTIFIER",
            "",
        ).strip()
        or platform.machine().strip()
    )

    if not cpu_model:
        raise RuntimeError(
            "Could not determine CPU model"
        )

    cpu_core_count = (
        os.cpu_count()
    )

    if (
        cpu_core_count is None
        or cpu_core_count <= 0
    ):
        raise RuntimeError(
            "Could not determine CPU core count"
        )

    cpu_architecture = (
        platform.machine().strip()
        or platform.architecture()[0].strip()
    )

    if not cpu_architecture:
        raise RuntimeError(
            "Could not determine CPU architecture"
        )

    os_identifier = (
        platform.platform().strip()
    )

    if not os_identifier:
        raise RuntimeError(
            "Could not determine OS identifier"
        )

    return ExecutionEnvironment(
        python_version=(
            platform.python_version()
        ),
        engine_version=engine_version,
        dependency_lockfile=(
            poetry_lock.name
        ),
        poetry_lock_hash=sha256_file(
            poetry_lock
        ),
        git_commit=git_commit,
        cpu_model=cpu_model,
        cpu_core_count=(
            cpu_core_count
        ),
        cpu_architecture=(
            cpu_architecture
        ),
        ram_bytes=ram_bytes,
        os_identifier=(
            os_identifier
        ),
    )


def save_experiment_config(
    config: ExperimentConfig,
    path: str | Path,
) -> None:
    """
    Persist a complete experiment configuration with integrity hash.
    """
    destination = Path(
        path
    )

    payload: dict[
        str,
        JSONValue,
    ] = {
        "file_version": (
            EXPERIMENT_CONFIG_FILE_VERSION
        ),
        "configuration_hash": (
            config.configuration_hash
        ),
        "configuration": (
            canonical_experiment_config_dict(
                config
            )
        ),
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    )

    destination.write_text(
        serialized + "\n",
        encoding="utf-8",
    )


def load_experiment_config(
    path: str | Path,
) -> ExperimentConfig:
    """
    Load and integrity-check a persisted experiment configuration.
    """
    source = Path(
        path
    )

    raw = json.loads(
        source.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            "Experiment configuration file must contain a JSON object"
        )

    file_version = raw.get(
        "file_version"
    )

    if (
        file_version
        != EXPERIMENT_CONFIG_FILE_VERSION
    ):
        raise ValueError(
            "Unsupported experiment configuration file version"
        )

    stored_hash = raw.get(
        "configuration_hash"
    )

    if not isinstance(
        stored_hash,
        str,
    ):
        raise ValueError(
            "configuration_hash is missing or invalid"
        )

    _validate_sha256(
        stored_hash,
        "configuration_hash",
    )

    configuration_value = raw.get(
        "configuration"
    )

    if not isinstance(
        configuration_value,
        dict,
    ):
        raise ValueError(
            "configuration is missing or invalid"
        )

    config = experiment_config_from_dict(
        configuration_value
    )

    actual_hash = (
        config.configuration_hash
    )

    if actual_hash != stored_hash:
        raise ValueError(
            "Experiment configuration hash mismatch"
        )

    return config


def experiment_config_from_dict(
    data: Mapping[str, Any],
) -> ExperimentConfig:
    """
    Reconstruct a frozen ExperimentConfig from its canonical representation.
    """
    game_data = _require_mapping(
        data,
        "game_config",
    )
    bot_data = _require_mapping(
        data,
        "bot",
    )
    opponent_data = _require_mapping(
        data,
        "opponent",
    )
    population_data = _require_mapping(
        data,
        "population",
    )
    rng_data = _require_mapping(
        data,
        "rng",
    )
    sample_data = _require_mapping(
        data,
        "sample_size",
    )
    precision_data = _require_mapping(
        data,
        "precision_target",
    )
    method_data = _require_mapping(
        data,
        "statistical_method",
    )
    calibration_data = _require_mapping(
        data,
        "calibration",
    )
    environment_data = _require_mapping(
        data,
        "environment",
    )

    power_raw = data.get(
        "power_target"
    )

    if power_raw is None:
        power_target = None
    else:
        if not isinstance(
            power_raw,
            Mapping,
        ):
            raise ValueError(
                "power_target must be an object or null"
            )

        power_target = (
            _power_target_from_dict(
                power_raw
            )
        )

    seeds_raw = data.get(
        "master_seeds"
    )

    if not isinstance(
        seeds_raw,
        list,
    ):
        raise ValueError(
            "master_seeds must be a list"
        )

    master_seeds: list[int] = []

    for seed in seeds_raw:
        if (
            not isinstance(
                seed,
                int,
            )
            or isinstance(
                seed,
                bool,
            )
        ):
            raise ValueError(
                "master_seeds must contain integers"
            )

        master_seeds.append(
            seed
        )

    adaptive_raw = data.get(
        "adaptive"
    )

    if not isinstance(
        adaptive_raw,
        bool,
    ):
        raise ValueError(
            "adaptive must be a boolean"
        )

    return ExperimentConfig(
        experiment_id=_required_string(
            data,
            "experiment_id",
        ),
        game_config=(
            _game_config_from_dict(
                game_data
            )
        ),
        bot=_agent_config_from_dict(
            bot_data
        ),
        opponent=(
            _agent_config_from_dict(
                opponent_data
            )
        ),
        population=(
            _population_config_from_dict(
                population_data
            )
        ),
        master_seeds=tuple(
            master_seeds
        ),
        rng=_rng_config_from_dict(
            rng_data
        ),
        adaptive=adaptive_raw,
        statistical_unit=StatisticalUnit(
            _required_string(
                data,
                "statistical_unit",
            )
        ),
        sample_size=(
            _sample_size_from_dict(
                sample_data
            )
        ),
        precision_target=(
            _precision_target_from_dict(
                precision_data
            )
        ),
        power_target=power_target,
        statistical_method=(
            _statistical_method_from_dict(
                method_data
            )
        ),
        calibration=(
            _calibration_from_dict(
                calibration_data
            )
        ),
        illegal_action_policy=(
            IllegalActionPolicy(
                _required_string(
                    data,
                    "illegal_action_policy",
                )
            )
        ),
        environment=(
            _environment_from_dict(
                environment_data
            )
        ),
        schema_version=(
            _required_string(
                data,
                "schema_version",
            )
        ),
    )


def _game_config_from_dict(
    data: Mapping[str, Any],
) -> GameConfig:
    bet_abstraction_raw = data.get(
        "bet_abstraction"
    )

    if not isinstance(
        bet_abstraction_raw,
        list,
    ):
        raise ValueError(
            "game_config.bet_abstraction must be a list"
        )

    allowed_actions_raw = data.get(
        "allowed_actions"
    )

    if not isinstance(
        allowed_actions_raw,
        list,
    ):
        raise ValueError(
            "game_config.allowed_actions must be a list"
        )

    return GameConfig(
        game_type=GameType(
            _required_string(
                data,
                "game_type",
            )
        ),
        player_count=_required_int(
            data,
            "player_count",
        ),
        starting_stack_bb=Decimal(
            _required_string(
                data,
                "starting_stack_bb",
            )
        ),
        small_blind_bb=Decimal(
            _required_string(
                data,
                "small_blind_bb",
            )
        ),
        big_blind_bb=Decimal(
            _required_string(
                data,
                "big_blind_bb",
            )
        ),
        ante_bb=Decimal(
            _required_string(
                data,
                "ante_bb",
            )
        ),
        rake=Decimal(
            _required_string(
                data,
                "rake",
            )
        ),
        stack_reset_bb=Decimal(
            _required_string(
                data,
                "stack_reset_bb",
            )
        ),
        bet_abstraction=tuple(
            Decimal(
                str(value)
            )
            for value in bet_abstraction_raw
        ),
        allowed_actions=tuple(
            Action(
                str(value)
            )
            for value in allowed_actions_raw
        ),
        abstraction_version=(
            _required_string(
                data,
                "abstraction_version",
            )
        ),
    )


def _agent_config_from_dict(
    data: Mapping[str, Any],
) -> AgentExperimentConfig:
    simulation_budget = data.get(
        "simulation_budget"
    )
    time_budget_ms = data.get(
        "time_budget_ms"
    )

    if (
        simulation_budget is not None
        and (
            not isinstance(
                simulation_budget,
                int,
            )
            or isinstance(
                simulation_budget,
                bool,
            )
        )
    ):
        raise ValueError(
            "simulation_budget must be an integer or null"
        )

    if (
        time_budget_ms is not None
        and (
            not isinstance(
                time_budget_ms,
                int,
            )
            or isinstance(
                time_budget_ms,
                bool,
            )
        )
    ):
        raise ValueError(
            "time_budget_ms must be an integer or null"
        )

    return AgentExperimentConfig(
        name=_required_string(
            data,
            "name",
        ),
        version=_required_string(
            data,
            "version",
        ),
        parameters=(
            CanonicalJSON.from_value(
                cast(
                    JSONValue,
                    data.get(
                        "parameters"
                    ),
                )
            )
        ),
        simulation_budget=(
            simulation_budget
        ),
        time_budget_ms=(
            time_budget_ms
        ),
        timeout_policy=(
            _required_string(
                data,
                "timeout_policy",
            )
        ),
    )


def _population_config_from_dict(
    data: Mapping[str, Any],
) -> PopulationConfig:
    population_seed = data.get(
        "population_seed"
    )

    if (
        population_seed is not None
        and (
            not isinstance(
                population_seed,
                int,
            )
            or isinstance(
                population_seed,
                bool,
            )
        )
    ):
        raise ValueError(
            "population_seed must be an integer or null"
        )

    return PopulationConfig(
        population_id=(
            _required_string(
                data,
                "population_id",
            )
        ),
        version=_required_string(
            data,
            "version",
        ),
        distribution=(
            CanonicalJSON.from_value(
                cast(
                    JSONValue,
                    data.get(
                        "distribution"
                    ),
                )
            )
        ),
        population_seed=(
            population_seed
        ),
    )


def _rng_config_from_dict(
    data: Mapping[str, Any],
) -> RNGConfiguration:
    return RNGConfiguration(
        version=_required_string(
            data,
            "version",
        ),
        crn_mode=CRNMode(
            _required_string(
                data,
                "crn_mode",
            )
        ),
        deal_namespace=(
            _required_string(
                data,
                "deal_namespace",
            )
        ),
        bot_a_namespace=(
            _required_string(
                data,
                "bot_a_namespace",
            )
        ),
        bot_b_namespace=(
            _required_string(
                data,
                "bot_b_namespace",
            )
        ),
        agent_event_key=(
            _required_string(
                data,
                "agent_event_key",
            )
        ),
    )


def _sample_size_from_dict(
    data: Mapping[str, Any],
) -> SampleSizePlan:
    return SampleSizePlan(
        planned_simulated_hands=(
            _required_int(
                data,
                "planned_simulated_hands",
            )
        ),
        planned_independent_units=(
            _required_int(
                data,
                "planned_independent_units",
            )
        ),
    )


def _power_target_from_dict(
    data: Mapping[str, Any],
) -> PowerTarget:
    return PowerTarget(
        alpha=_required_float(
            data,
            "alpha",
        ),
        beta=_required_float(
            data,
            "beta",
        ),
        minimum_relevant_difference_bb100=(
            _required_float(
                data,
                "minimum_relevant_difference_bb100",
            )
        ),
        estimated_sd_difference_bb100=(
            _required_float(
                data,
                "estimated_sd_difference_bb100",
            )
        ),
        variance_source=(
            _required_string(
                data,
                "variance_source",
            )
        ),
    )


def _precision_target_from_dict(
    data: Mapping[str, Any],
) -> PrecisionTarget:
    return PrecisionTarget(
        estimand=_required_string(
            data,
            "estimand",
        ),
        target_ci_half_width_bb100=(
            _required_float(
                data,
                "target_ci_half_width_bb100",
            )
        ),
        confidence_level=(
            _required_float(
                data,
                "confidence_level",
            )
        ),
        statistical_unit=(
            StatisticalUnit(
                _required_string(
                    data,
                    "statistical_unit",
                )
            )
        ),
        variance_estimator=(
            _required_string(
                data,
                "variance_estimator",
            )
        ),
    )


def _statistical_method_from_dict(
    data: Mapping[str, Any],
) -> StatisticalMethodConfig:
    return StatisticalMethodConfig(
        bootstrap_seed=(
            _required_int(
                data,
                "bootstrap_seed",
            )
        ),
        method_version=(
            _required_string(
                data,
                "method_version",
            )
        ),
        confidence_level=(
            _required_float(
                data,
                "confidence_level",
            )
        ),
        bootstrap_resamples=(
            _required_int(
                data,
                "bootstrap_resamples",
            )
        ),
        bootstrap_percentile_method=(
            _required_string(
                data,
                "bootstrap_percentile_method",
            )
        ),
        effect_size=(
            _required_string(
                data,
                "effect_size",
            )
        ),
    )


def _calibration_from_dict(
    data: Mapping[str, Any],
) -> CalibrationRecord:
    return CalibrationRecord(
        version=_required_string(
            data,
            "version",
        ),
        status=_required_string(
            data,
            "status",
        ),
        results=(
            CanonicalJSON.from_value(
                cast(
                    JSONValue,
                    data.get(
                        "results"
                    ),
                )
            )
        ),
    )


def _environment_from_dict(
    data: Mapping[str, Any],
) -> ExecutionEnvironment:
    return ExecutionEnvironment(
        python_version=(
            _required_string(
                data,
                "python_version",
            )
        ),
        engine_version=(
            _required_string(
                data,
                "engine_version",
            )
        ),
        dependency_lockfile=(
            _required_string(
                data,
                "dependency_lockfile",
            )
        ),
        poetry_lock_hash=(
            _required_string(
                data,
                "poetry_lock_hash",
            )
        ),
        git_commit=(
            _required_string(
                data,
                "git_commit",
            )
        ),
        cpu_model=_required_string(
            data,
            "cpu_model",
        ),
        cpu_core_count=(
            _required_int(
                data,
                "cpu_core_count",
            )
        ),
        cpu_architecture=(
            _required_string(
                data,
                "cpu_architecture",
            )
        ),
        ram_bytes=_required_int(
            data,
            "ram_bytes",
        ),
        os_identifier=(
            _required_string(
                data,
                "os_identifier",
            )
        ),
    )


def _require_mapping(
    data: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = data.get(
        key
    )

    if not isinstance(
        value,
        Mapping,
    ):
        raise ValueError(
            f"{key} must be an object"
        )

    return value


def _required_string(
    data: Mapping[str, Any],
    key: str,
) -> str:
    value = data.get(
        key
    )

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"{key} must be a string"
        )

    return value


def _required_int(
    data: Mapping[str, Any],
    key: str,
) -> int:
    value = data.get(
        key
    )

    if (
        not isinstance(
            value,
            int,
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise ValueError(
            f"{key} must be an integer"
        )

    return value


def _required_float(
    data: Mapping[str, Any],
    key: str,
) -> float:
    value = data.get(
        key
    )

    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise ValueError(
            f"{key} must be numeric"
        )

    converted = float(
        value
    )

    if not math.isfinite(
        converted
    ):
        raise ValueError(
            f"{key} must be finite"
        )

    return converted