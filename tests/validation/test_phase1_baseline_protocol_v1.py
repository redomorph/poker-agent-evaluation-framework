from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.experiments.final_protocol import (
    PHASE1_FINAL_TARGET_HALF_WIDTH_BB100,
    PHASE1_PILOT_REPORT_HASH,
    FinalProtocolError,
    iter_final_jobs,
    load_phase1_baseline_protocol,
    sha256_canonical,
    verify_pilot_report_binding,
)
from pokerbench.experiments.frozen_protocol import (
    canonical_file_bytes,
    load_frozen_population,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    ROOT / "configs" / "experiments" / "phase1-baseline-experiment-v1.json"
)
PRIMARY_PATH = ROOT / "configs" / "populations" / "baseline-synthetic-v1.json"
BALANCED_PATH = ROOT / "configs" / "populations" / "balanced-v1.json"


def _populations():
    primary = load_frozen_population(PRIMARY_PATH)
    balanced = load_frozen_population(BALANCED_PATH)
    return {
        primary.population_id: primary,
        balanced.population_id: balanced,
    }


def _protocol():
    return load_phase1_baseline_protocol(
        PROTOCOL_PATH,
        populations=_populations(),
    )


def _write_mutated_protocol(tmp_path: Path, mutator) -> Path:
    envelope = json.loads(PROTOCOL_PATH.read_bytes())
    mutator(envelope["protocol"])
    envelope["protocol_hash"] = sha256_canonical(envelope["protocol"])
    path = tmp_path / "mutated.json"
    path.write_bytes(canonical_file_bytes(envelope))
    return path


def test_final_protocol_file_is_canonical_and_hash_valid() -> None:
    raw = PROTOCOL_PATH.read_bytes()
    envelope = json.loads(raw)

    assert raw == canonical_file_bytes(envelope)
    assert envelope["protocol_hash"] == sha256_canonical(envelope["protocol"])

    protocol = _protocol()
    assert protocol.protocol_id == "phase1-baseline-experiment-v1"
    assert (
        protocol.protocol_hash
        == "82cfd5d0da934685af8be50a4f2ad13219c4c276534d206260e8479d8e242674"
    )


def test_precision_deviation_is_explicit_and_not_misrepresented() -> None:
    protocol = _protocol()
    envelope = json.loads(PROTOCOL_PATH.read_bytes())["protocol"]
    disclosure = envelope["precision"]["disclosure"]

    assert protocol.target_half_width_bb100 == PHASE1_FINAL_TARGET_HALF_WIDTH_BB100
    assert protocol.target_half_width_bb100 == 50.0
    assert protocol.illustrative_project_half_width_bb100 == 1.0
    assert disclosure["deviation_from_illustrative_example"] is True
    assert disclosure["deviation_type"] == "documented_coarser_precision_target"
    assert (
        disclosure["selection_basis"]
        == "completed_phase1_pilot_variance_and_runtime_before_final_data"
    )
    assert (
        disclosure["interpretation"]
        == "the_final_experiment_is_not_equivalent_to_plus_minus_1_bb100_precision"
    )


def test_final_sample_sizes_are_mechanically_derived_from_pilot_variance() -> None:
    protocol = _protocol()
    plans = {plan.agent_id: plan for plan in protocol.agent_plans}

    assert plans["random"].raw_required_units_per_profile == pytest.approx(
        2768.216219839214,
        abs=1e-12,
    )
    assert plans["random"].paired_units_per_profile == 2769
    assert plans["random"].planned_half_width_bb100 == pytest.approx(
        49.9929231163604,
        abs=1e-12,
    )

    assert plans["hand_strength"].raw_required_units_per_profile == pytest.approx(
        360.1581107091833,
        abs=1e-12,
    )
    assert plans["hand_strength"].paired_units_per_profile == 361
    assert plans["hand_strength"].planned_half_width_bb100 == pytest.approx(
        49.94166338592805,
        abs=1e-12,
    )

    assert protocol.planned_independent_units == 15_650
    assert protocol.planned_simulated_engine_hands == 31_300


def test_rehashed_sample_size_tampering_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_protocol(
        tmp_path,
        lambda payload: payload["hero_agents"][0].update(
            paired_units_per_profile=2768,
            seed_count=2768,
        ),
    )

    with pytest.raises(FinalProtocolError, match=r"ceil\(pilot precision requirement\)"):
        load_phase1_baseline_protocol(path, populations=_populations())


def test_rehashed_precision_target_tampering_is_rejected(tmp_path: Path) -> None:
    path = _write_mutated_protocol(
        tmp_path,
        lambda payload: payload["precision"].update(target_half_width_bb100=25.0),
    )

    with pytest.raises(FinalProtocolError, match="remain frozen at 95% and ±50"):
        load_phase1_baseline_protocol(path, populations=_populations())


def test_final_seed_schedules_are_fixed_disjoint_from_pilot_and_share_prefix() -> None:
    protocol = _protocol()
    plans = {plan.agent_id: plan for plan in protocol.agent_plans}
    pilot = set(range(6_100_000, 6_100_100))

    assert plans["random"].master_seeds[0] == 6_200_000
    assert plans["random"].master_seeds[-1] == 6_202_768
    assert plans["hand_strength"].master_seeds[0] == 6_200_000
    assert plans["hand_strength"].master_seeds[-1] == 6_200_360
    assert plans["random"].master_seeds[:361] == plans["hand_strength"].master_seeds
    assert pilot.isdisjoint(plans["random"].master_seeds)
    assert pilot.isdisjoint(plans["hand_strength"].master_seeds)


def test_final_job_expansion_matches_frozen_matrix() -> None:
    protocol = _protocol()
    jobs = iter_final_jobs(protocol)

    assert len(jobs) == 15_650
    assert jobs[0].cell_id == "random-vs-nit"
    assert jobs[0].master_seed == 6_200_000
    assert jobs[2768].master_seed == 6_202_768
    assert jobs[-1].cell_id == "hand_strength-vs-maniac"
    assert jobs[-1].master_seed == 6_200_360
    assert len(
        {
            (job.hero_id, job.opponent_profile_id, job.master_seed)
            for job in jobs
        }
    ) == len(jobs)


def test_final_phase1_is_precision_planned_not_confirmatory() -> None:
    payload = json.loads(PROTOCOL_PATH.read_bytes())["protocol"]

    assert payload["scientific_use"]["final_experiment_data"] is True
    assert payload["scientific_use"]["eligible_for_confirmatory_analysis"] is False
    assert payload["scientific_use"]["hypothesis_test_planned"] is False
    assert payload["scientific_use"]["optional_stopping"] is False
    assert payload["scientific_use"]["pilot_data_included_in_final_estimate"] is False
    assert payload["power"]["alpha"] is None
    assert payload["power"]["beta"] is None
    assert payload["power"]["minimum_relevant_difference_bb100"] is None


def test_pilot_and_gate6_evidence_are_hash_bound() -> None:
    protocol = _protocol()

    assert (
        protocol.pilot_protocol_hash
        == "cc589fc65610bca308d8ed739efab848fff8477e88c20043b045d5ad3ee6c0a9"
    )
    assert protocol.pilot_report_hash == PHASE1_PILOT_REPORT_HASH
    assert (
        protocol.gate6_report_hash
        == "ab6f95e46d7f0d6c1df63240e2473a102d8c1cbd582cc488acc026c6f618394a"
    )
    assert len(protocol.pilot_variance_cells) == 10
    assert all(cell.valid_units == 100 for cell in protocol.pilot_variance_cells)


def test_pilot_report_binding_verifier_accepts_matching_complete_evidence(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    report = {
        "report_version": "phase1-pilot-execution-report-v1",
        "protocol_hash": protocol.pilot_protocol_hash,
        "gate6_report_hash": protocol.gate6_report_hash,
        "complete": True,
        "failure_free": True,
        "completed_jobs": 1000,
        "valid_units": 1000,
        "failed_units": 0,
        "recorded_engine_hands": 2000,
        "scientific_use": {
            "contains_confirmatory_inference": False,
            "contains_final_benchmark_estimate": False,
            "final_sample_size_planning": True,
            "variance_estimation": True,
        },
        "cells": [
            {
                "hero_id": cell.hero_id,
                "opponent_profile_id": cell.opponent_profile_id,
                "std_difference_bb100": cell.std_difference_bb100,
                "valid_units": cell.valid_units,
            }
            for cell in protocol.pilot_variance_cells
        ],
    }
    report_hash = sha256_canonical(report)
    synthetic_protocol = replace(protocol, pilot_report_hash=report_hash)
    path = tmp_path / "pilot-report.json"
    path.write_bytes(
        canonical_file_bytes({"report": report, "report_hash": report_hash})
    )

    verify_pilot_report_binding(synthetic_protocol, path)


def test_pilot_report_binding_verifier_rejects_tampering(tmp_path: Path) -> None:
    protocol = _protocol()
    path = tmp_path / "pilot-report.json"
    path.write_bytes(
        canonical_file_bytes(
            {
                "report": {"report_version": "phase1-pilot-execution-report-v1"},
                "report_hash": "0" * 64,
            }
        )
    )

    with pytest.raises(FinalProtocolError, match="Pilot report hash"):
        verify_pilot_report_binding(protocol, path)


def test_scope_limitations_are_explicit_not_silently_claimed_complete() -> None:
    payload = json.loads(PROTOCOL_PATH.read_bytes())["protocol"]
    scope = payload["scope_limitations"]

    assert scope["complete_dtest_population_benchmark"] is False
    assert scope["meets_minimum_10_independent_opponent_samples"] is False
    assert scope["static_profile_count"] == 5
    assert scope["tail_precision_warning"] is True
    assert scope["full_section_11_2_benchmark_matrix"] is False
    assert scope["randomized_population_frozen"] is False
    assert scope["stress_population_frozen"] is False
    assert scope["holdout_population_frozen"] is False
