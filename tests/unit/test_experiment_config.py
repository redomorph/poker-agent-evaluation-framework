# tests/unit/test_experiment_config.py
"""Unit tests for frozen experiment configuration and persistence."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from pokerbench.engine.config import (
    BASELINE_GAME_CONFIG,
)
from pokerbench.experiments import (
    AgentExperimentConfig,
    CalibrationRecord,
    CanonicalJSON,
    CRNMode,
    ExecutionEnvironment,
    ExperimentConfig,
    PopulationConfig,
    PrecisionTarget,
    RNGConfiguration,
    SampleSizePlan,
    StatisticalMethodConfig,
    canonical_experiment_config_json,
    capture_execution_environment,
    experiment_config_hash,
    load_experiment_config,
    save_experiment_config,
    sha256_file,
)
from pokerbench.runner import (
    IllegalActionPolicy,
    StatisticalUnit,
)


TEST_SHA256 = "a" * 64


def _environment() -> ExecutionEnvironment:
    return ExecutionEnvironment(
        python_version="3.14.0",
        engine_version="0.1.0",
        dependency_lockfile="poetry.lock",
        poetry_lock_hash=TEST_SHA256,
        git_commit="0123456789abcdef",
        cpu_model="Test CPU",
        cpu_core_count=8,
        cpu_architecture="x86_64",
        ram_bytes=16 * 1024**3,
        os_identifier="Test OS",
    )


def _agent(
    name: str,
    version: str,
) -> AgentExperimentConfig:
    return AgentExperimentConfig(
        name=name,
        version=version,
        parameters=CanonicalJSON.from_value(
            {
                "policy": "phase0-symmetry-v1",
                "fold_probability": 0.005,
            }
        ),
        simulation_budget=None,
        time_budget_ms=None,
        timeout_policy="fold",
    )


def _config(
    *,
    master_seeds: tuple[int, ...] = (
        42,
        12_345,
        999,
    ),
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id="phase0-symmetry-001",
        game_config=BASELINE_GAME_CONFIG,
        bot=_agent(
            "A",
            "phase0-symmetry-v1",
        ),
        opponent=_agent(
            "B",
            "phase0-symmetry-v1",
        ),
        population=PopulationConfig(
            population_id="phase0-identical-random",
            version="v1",
            distribution=CanonicalJSON.from_value(
                {
                    "type": "fixed_identical_policy",
                }
            ),
            population_seed=None,
        ),
        master_seeds=master_seeds,
        rng=RNGConfiguration(
            version="event-indexed-v1",
            crn_mode=(
                CRNMode.IDENTICAL_REALISATIONS
            ),
        ),
        adaptive=False,
        statistical_unit=(
            StatisticalUnit.PAIRED_DEAL
        ),
        sample_size=SampleSizePlan(
            # 3 seeds * 5000 paired deals * 2 positions.
            planned_simulated_hands=30_000,
            planned_independent_units=15_000,
        ),
        precision_target=PrecisionTarget(
            estimand=(
                "position-controlled A-vs-B "
                "paired mean difference"
            ),
            target_ci_half_width_bb100=0.5,
            confidence_level=0.95,
            statistical_unit=(
                StatisticalUnit.PAIRED_DEAL
            ),
            variance_estimator=(
                "phase0-symmetry-v1 "
                "conservative variance bound"
            ),
        ),
        power_target=None,
        statistical_method=(
            StatisticalMethodConfig(
                bootstrap_seed=20260810,
            )
        ),
        calibration=CalibrationRecord(
            version="phase0-calibration-v1",
            status="not_applicable",
            results=CanonicalJSON.from_value(
                {
                    "reason": (
                        "Phase-0 symmetry validation "
                        "policy is not an opponent "
                        "population profile"
                    )
                }
            ),
        ),
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
        environment=_environment(),
    )


def test_canonical_json_normalizes_key_order(
) -> None:
    first = CanonicalJSON(
        '{"b":2,"a":1}'
    )

    second = CanonicalJSON(
        '{ "a": 1, "b": 2 }'
    )

    assert first == second

    assert (
        first.text
        == '{"a":1,"b":2}'
    )


def test_experiment_config_is_frozen(
) -> None:
    config = _config()

    with pytest.raises(
        dataclasses.FrozenInstanceError
    ):
        config.experiment_id = "changed"  # type: ignore[misc]


def test_statistical_unit_is_explicit_and_matches_precision_target(
) -> None:
    config = _config()

    assert (
        config.statistical_unit
        == StatisticalUnit.PAIRED_DEAL
    )

    assert (
        config.precision_target.statistical_unit
        == config.statistical_unit
    )


def test_mismatched_precision_statistical_unit_is_rejected(
) -> None:
    with pytest.raises(
        ValueError,
        match="statistical unit",
    ):
        ExperimentConfig(
            experiment_id="invalid",
            game_config=BASELINE_GAME_CONFIG,
            bot=_agent(
                "A",
                "v1",
            ),
            opponent=_agent(
                "B",
                "v1",
            ),
            population=PopulationConfig(
                population_id="test",
                version="v1",
                distribution=(
                    CanonicalJSON.from_value(
                        {}
                    )
                ),
                population_seed=None,
            ),
            master_seeds=(1,),
            rng=RNGConfiguration(
                version="event-indexed-v1",
                crn_mode=(
                    CRNMode.IDENTICAL_REALISATIONS
                ),
            ),
            adaptive=False,
            statistical_unit=(
                StatisticalUnit.PAIRED_DEAL
            ),
            sample_size=SampleSizePlan(
                planned_simulated_hands=20,
                planned_independent_units=10,
            ),
            precision_target=PrecisionTarget(
                estimand="test",
                target_ci_half_width_bb100=1.0,
                confidence_level=0.95,
                statistical_unit=(
                    StatisticalUnit.TRAJECTORY_CLUSTER
                ),
                variance_estimator="pilot",
            ),
            power_target=None,
            statistical_method=(
                StatisticalMethodConfig(
                    bootstrap_seed=1,
                )
            ),
            calibration=CalibrationRecord(
                version="v1",
                status="not_applicable",
                results=(
                    CanonicalJSON.from_value(
                        {}
                    )
                ),
            ),
            illegal_action_policy=(
                IllegalActionPolicy.EXCLUDE_RUN
            ),
            environment=_environment(),
        )


def test_adaptive_experiment_rejects_paired_deal_unit(
) -> None:
    original = _config()

    with pytest.raises(
        ValueError,
        match="Adaptive",
    ):
        dataclasses.replace(
            original,
            adaptive=True,
        )


def test_configuration_hash_is_deterministic(
) -> None:
    first = _config()
    second = _config()

    assert (
        canonical_experiment_config_json(
            first
        )
        == canonical_experiment_config_json(
            second
        )
    )

    assert (
        experiment_config_hash(
            first
        )
        == experiment_config_hash(
            second
        )
    )

    assert (
        first.configuration_hash
        == second.configuration_hash
    )


def test_configuration_hash_changes_when_seed_changes(
) -> None:
    first = _config(
        master_seeds=(
            42,
            12_345,
            999,
        )
    )

    second = _config(
        master_seeds=(
            42,
            12_345,
            1_000,
        )
    )

    assert (
        first.configuration_hash
        != second.configuration_hash
    )


def test_save_and_load_round_trip(
    tmp_path: Path,
) -> None:
    config = _config()

    path = (
        tmp_path
        / "experiment.json"
    )

    save_experiment_config(
        config,
        path,
    )

    loaded = (
        load_experiment_config(
            path
        )
    )

    assert loaded == config

    assert (
        loaded.configuration_hash
        == config.configuration_hash
    )


def test_tampered_persisted_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    config = _config()

    path = (
        tmp_path
        / "experiment.json"
    )

    save_experiment_config(
        config,
        path,
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    payload[
        "configuration"
    ][
        "master_seeds"
    ][0] = 123456

    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="hash mismatch",
    ):
        load_experiment_config(
            path
        )


def test_sha256_file_hashes_exact_bytes(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "poetry.lock"
    )

    content = b"exact lock state\n"

    path.write_bytes(
        content
    )

    expected = hashlib.sha256(
        content
    ).hexdigest()

    assert (
        sha256_file(path)
        == expected
    )


def test_capture_execution_environment_records_lock_state(
    tmp_path: Path,
) -> None:
    poetry_lock = (
        tmp_path
        / "poetry.lock"
    )

    poetry_lock.write_text(
        "test-lock-content",
        encoding="utf-8",
    )

    environment = (
        capture_execution_environment(
            engine_version="0.1.0",
            git_commit="abcdef1234567890",
            poetry_lock_path=poetry_lock,
            ram_bytes=16 * 1024**3,
        )
    )

    assert (
        environment.dependency_lockfile
        == "poetry.lock"
    )

    assert (
        environment.poetry_lock_hash
        == sha256_file(
            poetry_lock
        )
    )

    assert environment.python_version
    assert environment.cpu_model

    assert (
        environment.cpu_core_count
        > 0
    )

    assert environment.cpu_architecture
    assert environment.os_identifier