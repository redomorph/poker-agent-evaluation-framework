# tests/validation/test_experiment_config_gates.py
"""
Phase-0 gates for frozen experiment configuration and persistence.

These tests verify that:

1. the statistical unit is explicitly persisted before simulation;
2. the required CRN/RNG definition is persisted;
3. the complete reproducibility metadata is present;
4. canonical persistence is deterministic and integrity checked.
"""

from __future__ import annotations

from pathlib import Path

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
    canonical_experiment_config_dict,
    load_experiment_config,
    save_experiment_config,
)
from pokerbench.runner import (
    IllegalActionPolicy,
    StatisticalUnit,
)


PHASE0_SEEDS = (
    42,
    12_345,
    999,
)

PHASE0_PAIRS_PER_SEED = 5_000

PHASE0_TEST_LOCK_HASH = (
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
    "0123456789abcdef"
)


def _phase0_config() -> ExperimentConfig:
    agent_parameters = (
        CanonicalJSON.from_value(
            {
                "policy_version": (
                    "phase0-symmetry-v1"
                ),
                "fold_probability": 0.005,
                "aggressive_action_probability": 0.0,
            }
        )
    )

    return ExperimentConfig(
        experiment_id=(
            "phase0-symmetry-validation-v1"
        ),
        game_config=BASELINE_GAME_CONFIG,
        bot=AgentExperimentConfig(
            name="Phase0Random-A",
            version="phase0-symmetry-v1",
            parameters=agent_parameters,
            simulation_budget=None,
            time_budget_ms=None,
            timeout_policy="fold",
        ),
        opponent=AgentExperimentConfig(
            name="Phase0Random-B",
            version="phase0-symmetry-v1",
            parameters=agent_parameters,
            simulation_budget=None,
            time_budget_ms=None,
            timeout_policy="fold",
        ),
        population=PopulationConfig(
            population_id=(
                "phase0-identical-random"
            ),
            version="v1",
            distribution=(
                CanonicalJSON.from_value(
                    {
                        "type": (
                            "fixed_identical_policy"
                        )
                    }
                )
            ),
            population_seed=None,
        ),
        master_seeds=PHASE0_SEEDS,
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
            planned_simulated_hands=(
                len(PHASE0_SEEDS)
                * PHASE0_PAIRS_PER_SEED
                * 2
            ),
            planned_independent_units=(
                len(PHASE0_SEEDS)
                * PHASE0_PAIRS_PER_SEED
            ),
        ),
        precision_target=PrecisionTarget(
            estimand=(
                "position-controlled A-vs-B "
                "mean paired difference"
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
                bootstrap_resamples=10_000,
            )
        ),
        calibration=CalibrationRecord(
            version="phase0-validation-v1",
            status="not_applicable",
            results=(
                CanonicalJSON.from_value(
                    {
                        "reason": (
                            "Dedicated symmetry "
                            "validation policy"
                        )
                    }
                )
            ),
        ),
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
        environment=ExecutionEnvironment(
            python_version="3.14",
            engine_version="0.1.0",
            dependency_lockfile=(
                "poetry.lock"
            ),
            poetry_lock_hash=(
                PHASE0_TEST_LOCK_HASH
            ),
            git_commit=(
                "phase0-test-commit"
            ),
            cpu_model="validation-cpu",
            cpu_core_count=1,
            cpu_architecture="validation-arch",
            ram_bytes=1,
            os_identifier="validation-os",
        ),
    )


def test_statistical_unit_is_explicitly_persisted(
) -> None:
    config = _phase0_config()

    persisted = (
        canonical_experiment_config_dict(
            config
        )
    )

    assert (
        persisted["statistical_unit"]
        == "paired_deal"
    )

    precision = persisted[
        "precision_target"
    ]

    assert isinstance(
        precision,
        dict,
    )

    assert (
        precision["statistical_unit"]
        == "paired_deal"
    )


def test_phase0_rng_crn_contract_is_persisted(
) -> None:
    config = _phase0_config()

    persisted = (
        canonical_experiment_config_dict(
            config
        )
    )

    rng = persisted["rng"]

    assert isinstance(
        rng,
        dict,
    )

    assert (
        rng["crn_mode"]
        == "identical_realisations"
    )

    assert (
        rng["deal_namespace"]
        == "DEAL_SCHEDULE"
    )

    assert (
        rng["bot_a_namespace"]
        == "RNG_BOT_A"
    )

    assert (
        rng["bot_b_namespace"]
        == "RNG_BOT_B"
    )

    assert (
        rng["agent_event_key"]
        == (
            "(bot_id,hand_id,"
            "decision_index,purpose)"
        )
    )


def test_complete_reproducibility_metadata_is_present(
) -> None:
    config = _phase0_config()

    persisted = (
        canonical_experiment_config_dict(
            config
        )
    )

    required_top_level_fields = {
        "schema_version",
        "experiment_id",
        "game_config",
        "bot",
        "opponent",
        "population",
        "master_seeds",
        "rng",
        "adaptive",
        "statistical_unit",
        "sample_size",
        "precision_target",
        "power_target",
        "statistical_method",
        "calibration",
        "illegal_action_policy",
        "environment",
    }

    assert (
        required_top_level_fields
        <= set(persisted)
    )

    environment = persisted[
        "environment"
    ]

    assert isinstance(
        environment,
        dict,
    )

    required_environment_fields = {
        "python_version",
        "engine_version",
        "dependency_lockfile",
        "poetry_lock_hash",
        "git_commit",
        "cpu_model",
        "cpu_core_count",
        "cpu_architecture",
        "ram_bytes",
        "os_identifier",
    }

    assert (
        required_environment_fields
        <= set(environment)
    )

    game_config = persisted[
        "game_config"
    ]

    assert isinstance(
        game_config,
        dict,
    )

    # The game object itself contains the bet abstraction and its
    # version; they are not represented by an ambiguous game-name-only
    # identifier.
    assert (
        game_config["bet_abstraction"]
        == [
            "0.33",
            "0.66",
            "1.0",
            "2.0",
        ]
    )

    assert (
        game_config[
            "abstraction_version"
        ]
        == "baseline-v1"
    )

    sample_size = persisted[
        "sample_size"
    ]

    assert isinstance(
        sample_size,
        dict,
    )

    assert (
        sample_size[
            "planned_independent_units"
        ]
        == 15_000
    )

    assert (
        sample_size[
            "planned_simulated_hands"
        ]
        == 30_000
    )


def test_persisted_experiment_round_trip_preserves_hash(
    tmp_path: Path,
) -> None:
    config = _phase0_config()

    path = (
        tmp_path
        / "phase0-experiment.json"
    )

    save_experiment_config(
        config,
        path,
    )

    first_bytes = (
        path.read_bytes()
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

    # Re-persisting an identical frozen experiment definition must
    # produce byte-identical canonical output.
    save_experiment_config(
        loaded,
        path,
    )

    second_bytes = (
        path.read_bytes()
    )

    assert (
        first_bytes
        == second_bytes
    )