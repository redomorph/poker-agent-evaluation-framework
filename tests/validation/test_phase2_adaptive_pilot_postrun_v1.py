from __future__ import annotations

import hashlib
import json
import statistics
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.agents.adaptive_lifecycle import AdaptiveHandStrengthAgent
from pokerbench.agents.base import Agent
from pokerbench.evaluation.calibration import BehaviorCounts
from pokerbench.evaluation.population import (
    BALANCED_STATIC_POPULATION,
    BASELINE_SYNTHETIC_STATIC_POPULATION,
)
from pokerbench.experiments.adaptive_pilot_analysis import TrajectoryArmEvidence
from pokerbench.experiments.adaptive_pilot_execution import AdaptivePilotPairExecution
from pokerbench.experiments.adaptive_pilot_postrun import (
    ADAPTIVE_PILOT_POSTRUN_ANALYSIS_VERSION,
    ADAPTIVE_PILOT_POSTRUN_REPORT_VERSION,
    AdaptivePilotCampaignAudit,
    AdaptivePilotLearningHorizonDiagnostic,
    AdaptivePilotLearningMetricDiagnostic,
    AdaptivePilotPairAuditEvidence,
    AdaptivePilotPostRunError,
    analyze_adaptive_pilot_postrun,
    build_adaptive_pilot_postrun_report,
    learning_horizon_diagnostics,
    load_adaptive_pilot_postrun_report,
    save_adaptive_pilot_postrun_report,
    _audit_completed_campaign,
)
from pokerbench.experiments.adaptive_pilot_protocol import iter_adaptive_pilot_jobs, load_adaptive_pilot_protocol
from pokerbench.replay.archive import (
    HandTermination,
    MatchRecord,
    RecordedAdaptiveLearningEvent,
    RecordedHand,
)
from pokerbench.engine.config import Action, BASELINE_GAME_CONFIG
from pokerbench.runner import PairedRunner


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-v1.json"

PROFILES = ("nit", "tag", "lag", "calling_station", "maniac")
SEEDS = (6_300_000, 6_300_001, 6_300_002, 6_300_003, 6_300_004)
METRICS = (
    "vpip",
    "pfr",
    "three_bet_freq",
    "cbet_freq",
    "postflop_aggressive_share",
    "fold_to_cbet",
)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _metric(metric: str, horizon: int) -> AdaptivePilotLearningMetricDiagnostic:
    return AdaptivePilotLearningMetricDiagnostic(
        metric=metric,
        opportunities=horizon,
        events=horizon // 2,
        alpha=1 + horizon // 2,
        beta=1 + horizon - horizon // 2,
    )


def _diagnostics() -> tuple[AdaptivePilotLearningHorizonDiagnostic, ...]:
    items: list[AdaptivePilotLearningHorizonDiagnostic] = []
    for profile in PROFILES:
        for seed in SEEDS:
            horizons = (10, 20, 50, 100, 500) if seed == 6_300_000 else (10, 20, 50, 100)
            for config_id in ("A_BTN", "B_BTN"):
                for horizon in horizons:
                    items.append(
                        AdaptivePilotLearningHorizonDiagnostic(
                            profile_id=profile,
                            master_seed=seed,
                            config_id=config_id,
                            horizon=horizon,
                            trajectory_hand_count=horizon,
                            valid_learning_transition_count=horizon,
                            posterior_state_hash=_hash(f"state:{profile}:{seed}:{config_id}:{horizon}"),
                            metrics=tuple(_metric(metric, horizon) for metric in METRICS),
                        )
                    )
    return tuple(items)


def _pair_execution(profile_index: int, profile: str, seed_index: int, seed: int) -> AdaptivePilotPairExecution:
    hands = 500 if seed == 6_300_000 else 100
    control_bb100 = 2.0 * profile_index
    difference_bb100 = 1.0 + profile_index + 0.5 * seed_index
    treatment_bb100 = control_bb100 + difference_bb100
    control_profit = control_bb100 / 100.0
    treatment_profit = treatment_bb100 / 100.0
    return AdaptivePilotPairExecution(
        profile_id=profile,
        master_seed=seed,
        hands_per_position=hands,
        scientifically_valid=True,
        treatment_match_id=f"t-{profile}-{seed}",
        control_match_id=f"c-{profile}-{seed}",
        treatment_record_hash=_hash(f"tr:{profile}:{seed}"),
        control_record_hash=_hash(f"cr:{profile}:{seed}"),
        treatment_failed_runs=(),
        control_failed_runs=(),
        treatment_learning_hands_by_position=(("A_BTN", hands), ("B_BTN", hands)),
        treatment_evidence=TrajectoryArmEvidence(
            arm_id="treatment",
            profile_id=profile,
            master_seed=seed,
            hero_btn_profit_bb=(treatment_profit,) * hands,
            hero_bb_profit_bb=(treatment_profit,) * hands,
        ),
        control_evidence=TrajectoryArmEvidence(
            arm_id="control",
            profile_id=profile,
            master_seed=seed,
            hero_btn_profit_bb=(control_profit,) * hands,
            hero_bb_profit_bb=(control_profit,) * hands,
        ),
    )


def _audit() -> AdaptivePilotCampaignAudit:
    pairs: list[AdaptivePilotPairExecution] = []
    evidence: list[AdaptivePilotPairAuditEvidence] = []
    for profile_index, profile in enumerate(PROFILES):
        for seed_index, seed in enumerate(SEEDS):
            pair = _pair_execution(profile_index, profile, seed_index, seed)
            pairs.append(pair)
            evidence.append(
                AdaptivePilotPairAuditEvidence(
                    profile_id=profile,
                    master_seed=seed,
                    hands_per_position=pair.hands_per_position,
                    treatment_match_id=pair.treatment_match_id,
                    control_match_id=pair.control_match_id,
                    treatment_record_hash=pair.treatment_record_hash,
                    control_record_hash=pair.control_record_hash,
                    treatment_snapshot_hash=_hash(f"ts:{profile}:{seed}"),
                    control_snapshot_hash=_hash(f"cs:{profile}:{seed}"),
                    pair_checkpoint_hash=_hash(f"pc:{profile}:{seed}"),
                )
            )
    return AdaptivePilotCampaignAudit(
        binding_hash=_hash("binding"),
        manifest_hash=_hash("manifest"),
        terminal_jobs=50,
        completed_jobs=50,
        failed_jobs=0,
        valid_pairs=25,
        invalid_pairs=0,
        recorded_engine_hands=18_000,
        total_arm_runtime_ns=50_000,
        median_arm_runtime_ns=1_000.0,
        database_bytes=123_456,
        work_root_bytes=654_321,
        replayed_treatment_match_records=25,
        replayed_learning_transitions=9_000,
        pair_evidence=tuple(evidence),
        pair_executions=tuple(pairs),
        learning_diagnostics=_diagnostics(),
    )


def _report():
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    return build_adaptive_pilot_postrun_report(
        protocol=protocol,
        execution_contract_id="phase2-adaptive-pilot-parallel-execution-v1",
        execution_contract_version="phase2-adaptive-pilot-parallel-execution-v1",
        execution_contract_hash=_hash("contract"),
        audit=_audit(),
        primary_population=BASELINE_SYNTHETIC_STATIC_POPULATION,
        secondary_population=BALANCED_STATIC_POPULATION,
    )


def test_report_covers_exact_frozen_windows_without_confirmatory_inference() -> None:
    report = _report()
    assert report.report_version == ADAPTIVE_PILOT_POSTRUN_REPORT_VERSION
    assert report.analysis_version == ADAPTIVE_PILOT_POSTRUN_ANALYSIS_VERSION
    assert report.complete and report.failure_free
    assert len(report.pair_evidence) == 25
    assert len(report.learning_diagnostics) == 210
    assert {item.window_id for item in report.profile_summaries} == {
        "cumulative_10", "cumulative_20", "cumulative_50", "cumulative_100", "cumulative_500",
        "early_adaptation", "main_adaptation", "steady_state",
    }
    assert report.scientific_use["confirmatory_h4_inference"] is False
    assert report.scientific_use["final_h4_effect_estimate"] is False
    assert report.scientific_use["pilot_plus_final_data_pooling"] is False
    assert report.final_h4_design == {
        "primary_h4_horizon": None,
        "final_h4_population": None,
        "minimum_relevant_difference_bb100": None,
        "precision_target_half_width_bb100": None,
        "final_independent_trajectory_clusters": None,
        "confirmatory_alpha": None,
        "confirmatory_beta": None,
        "multiplicity_binding": None,
        "final_h4_seed_schedule": None,
    }


def test_profile_cumulative_summary_uses_seed_level_contrasts_and_ddof1() -> None:
    report = _report()
    item = next(
        summary for summary in report.profile_summaries
        if summary.scope_id == "nit" and summary.window_id == "cumulative_100"
    )
    expected = [1.0 + 0.5 * index for index in range(5)]
    assert item.num_seed_contrasts == 5
    assert item.master_seeds == SEEDS
    assert item.mean_difference_bb100 == pytest.approx(statistics.fmean(expected))
    assert item.std_difference_bb100 == pytest.approx(statistics.stdev(expected))
    assert item.variance_estimator is not None


def test_500_and_steady_state_are_descriptive_only() -> None:
    report = _report()
    for window_id in ("cumulative_500", "steady_state"):
        items = [item for item in report.profile_summaries if item.window_id == window_id]
        assert len(items) == 5
        assert all(item.num_seed_contrasts == 1 for item in items)
        assert all(item.std_difference_bb100 is None for item in items)
        assert all(item.variance_estimator is None for item in items)
        assert all(item.inferential_statistics_permitted is False for item in items)


def test_population_summary_weights_profiles_within_seed_before_across_seed_sd() -> None:
    report = _report()
    item = next(
        summary for summary in report.population_summaries
        if summary.scope_id == "baseline-synthetic-static-profiles"
        and summary.window_id == "cumulative_100"
    )
    profile_component = 0.1 * 0 + 0.35 * 1 + 0.30 * 2 + 0.15 * 3 + 0.10 * 4
    expected_seed_values = [1.0 + profile_component + 0.5 * index for index in range(5)]
    assert item.num_seed_contrasts == 5
    assert item.mean_difference_bb100 == pytest.approx(statistics.fmean(expected_seed_values))
    assert item.std_difference_bb100 == pytest.approx(statistics.stdev(expected_seed_values))


def test_learning_diagnostics_have_exact_six_metrics_and_frozen_horizon_shape() -> None:
    report = _report()
    assert len(report.learning_diagnostics) == 210
    long = [
        item for item in report.learning_diagnostics
        if item.profile_id == "nit" and item.master_seed == 6_300_000 and item.config_id == "A_BTN"
    ]
    assert [item.horizon for item in long] == [10, 20, 50, 100, 500]
    assert all(tuple(metric.metric for metric in item.metrics) == METRICS for item in long)
    assert all(item.valid_learning_transition_count == item.horizon for item in long)


class _CallCheckAgent(Agent):
    def act(self, observation, legal_actions, rng):  # type: ignore[no-untyped-def]
        del observation, rng
        if Action.CHECK in legal_actions:
            return Action.CHECK
        if Action.CALL in legal_actions:
            return Action.CALL
        return Action.FOLD


class _ProbeAdaptiveAgent(_CallCheckAgent):
    def __init__(self) -> None:
        self._model = AdaptiveHandStrengthAgent.initial_model()

    @property
    def is_adaptive(self) -> bool:
        return True

    def reset(self) -> None:
        self._model = AdaptiveHandStrengthAgent.initial_model()

    def observe_opponent_hand(
        self, counts: BehaviorCounts, *, learner_bot_id: int, opponent_bot_id: int, trajectory_hand_index: int
    ) -> RecordedAdaptiveLearningEvent:
        before = self._model
        after = before.updated(counts, first_hand_index=before.hands_observed)
        event = RecordedAdaptiveLearningEvent.from_transition(
            learner_bot_id=learner_bot_id, opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index, before=before, evidence=counts, after=after,
        )
        self._model = after
        return event


def _learning_record(num_hands: int = 20) -> MatchRecord:
    result = PairedRunner(
        agent_a=_ProbeAdaptiveAgent(), agent_b=_CallCheckAgent(), num_hands=num_hands,
        seed=6_300_000, game_config=BASELINE_GAME_CONFIG, adaptive=True,
    ).run()
    assert result.match_record is not None
    return result.match_record


def test_learning_horizon_diagnostics_replays_public_chain_and_reports_state() -> None:
    diagnostics = learning_horizon_diagnostics(
        _learning_record(), profile_id="nit", master_seed=6_300_000, horizons=(10, 20)
    )
    assert [item.horizon for item in diagnostics] == [10, 20, 10, 20]
    assert [item.config_id for item in diagnostics] == ["A_BTN", "A_BTN", "B_BTN", "B_BTN"]
    assert diagnostics[-1].trajectory_hand_count == 20
    assert diagnostics[-1].valid_learning_transition_count == 20
    assert diagnostics[-1].posterior_state_hash == diagnostics[-1].posterior_state_hash.lower()
    assert {metric.metric for metric in diagnostics[-1].metrics} == set(METRICS)


def test_postrun_report_is_canonical_idempotent_and_rehashed_semantic_tamper_rejected(tmp_path: Path) -> None:
    report = _report()
    path = tmp_path / "report.json"
    save_adaptive_pilot_postrun_report(report, path)
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n") and b"\r" not in raw
    assert load_adaptive_pilot_postrun_report(path).report_hash == report.report_hash
    assert save_adaptive_pilot_postrun_report(report, path) == path

    document = json.loads(raw)
    document["report"]["scientific_use"]["confirmatory_h4_inference"] = True
    encoded_payload = json.dumps(
        document["report"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    document["report_hash"] = hashlib.sha256(encoded_payload).hexdigest()
    path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    )
    with pytest.raises(AdaptivePilotPostRunError, match="confirmatory"):
        load_adaptive_pilot_postrun_report(path)


def test_report_rejects_incomplete_or_invalid_campaign_evidence() -> None:
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    with pytest.raises(AdaptivePilotPostRunError, match="50 completed"):
        build_adaptive_pilot_postrun_report(
            protocol=protocol,
            execution_contract_id="x",
            execution_contract_version="y",
            execution_contract_hash=_hash("contract"),
            audit=replace(_audit(), completed_jobs=49, terminal_jobs=49),
            primary_population=BASELINE_SYNTHETIC_STATIC_POPULATION,
            secondary_population=BALANCED_STATIC_POPULATION,
        )
    with pytest.raises(AdaptivePilotPostRunError, match="25 valid"):
        build_adaptive_pilot_postrun_report(
            protocol=protocol,
            execution_contract_id="x",
            execution_contract_version="y",
            execution_contract_hash=_hash("contract"),
            audit=replace(_audit(), valid_pairs=24, invalid_pairs=1),
            primary_population=BASELINE_SYNTHETIC_STATIC_POPULATION,
            secondary_population=BALANCED_STATIC_POPULATION,
        )


def test_report_accounts_for_exact_18000_hands_and_9000_replayed_learning_transitions() -> None:
    report = _report()
    assert report.recorded_engine_hands == 18_000
    assert report.replayed_treatment_match_records == 25
    assert report.replayed_learning_transitions == 9_000
    assert report.failed_jobs == 0 and report.invalid_pairs == 0
    assert report.database_bytes == 123_456
    assert report.work_root_bytes == 654_321


def test_analyze_entrypoint_is_postrun_only_and_returns_saved_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = _report()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_postrun._analyze_from_paths",
        lambda **kwargs: expected,
    )
    path = tmp_path / "report.json"
    result = analyze_adaptive_pilot_postrun(
        protocol_path=PROTOCOL_PATH,
        execution_contract_path=ROOT / "configs" / "experiments" / "phase2-adaptive-pilot-parallel-execution-v1.json",
        primary_population_path=ROOT / "configs" / "populations" / "baseline-synthetic-v1.json",
        secondary_population_path=ROOT / "configs" / "populations" / "balanced-v1.json",
        database=tmp_path / "pilot.sqlite",
        work_root=tmp_path / "work",
        report_path=path,
    )
    assert result == expected
    assert path.exists()


def test_report_contains_no_p_values_confidence_intervals_or_final_h4_effect_estimate() -> None:
    payload = _report().payload()
    encoded = json.dumps(payload, sort_keys=True)
    assert "p_value" not in encoded
    assert "confidence_interval" not in encoded
    assert "ci_low" not in encoded and "ci_high" not in encoded
    assert payload["scientific_use"]["final_h4_effect_estimate"] is False


class _FakeAuditStore:
    def __init__(self, ids: tuple[str, ...], records: dict[str, object]) -> None:
        self._ids = ids
        self._records = records

    def list_match_ids(self) -> tuple[str, ...]:
        return self._ids

    def load_match(self, match_id: str):  # type: ignore[no-untyped-def]
        return self._records[match_id]


def _exercise_campaign_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, extra_store_id: bool = False):
    protocol = load_adaptive_pilot_protocol(PROTOCOL_PATH)
    jobs = iter_adaptive_pilot_jobs(protocol)
    binding = SimpleNamespace(binding_hash=_hash("binding"))
    states = tuple(
        SimpleNamespace(status="completed", match_id=f"match-{index}")
        for index, _ in enumerate(jobs)
    )
    manifest = SimpleNamespace(
        terminal_jobs=50,
        jobs=states,
        manifest_hash=_hash("manifest"),
    )
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_postrun.load_adaptive_pilot_run_manifest",
        lambda *args, **kwargs: manifest,
    )

    snapshot_dir = tmp_path / "work" / "arm-snapshots"
    checkpoint_dir = tmp_path / "work" / "pair-checkpoints"
    snapshot_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    job_index = {(job.arm_id, job.opponent_profile_id, job.master_seed): index for index, job in enumerate(jobs)}
    records: dict[str, object] = {}
    snapshots: dict[tuple[str, str, int], object] = {}
    for job in jobs:
        index = job_index[(job.arm_id, job.opponent_profile_id, job.master_seed)]
        (snapshot_dir / f"{job.arm_id}__{job.opponent_profile_id}__{job.master_seed}.json").write_text("{}")
        record_hash = _hash(f"record-{index}")
        if job.arm_id == "treatment":
            hand = SimpleNamespace(adaptive_learning_events=(SimpleNamespace(learner_bot_id=0),))
            hands = (hand,) * (2 * job.hands_per_position)
        else:
            hands = ()
        record = SimpleNamespace(record_hash=record_hash, hands=hands)
        records[f"match-{index}"] = record
        snapshots[(job.arm_id, job.opponent_profile_id, job.master_seed)] = SimpleNamespace(
            status="completed",
            match_id=f"match-{index}",
            match_record_hash=record_hash,
            recorded_engine_hands=2 * job.hands_per_position,
            elapsed_ns=100 + index,
            snapshot_hash=_hash(f"snapshot-{index}"),
            match_record=record,
            to_paired_result=lambda: object(),
        )
    for profile in PROFILES:
        for seed in SEEDS:
            (checkpoint_dir / f"{profile}__{seed}.json").write_text("{}")

    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_postrun.load_adaptive_pilot_arm_snapshot",
        lambda path, *, protocol, expected_binding, expected_job: snapshots[
            (expected_job.arm_id, expected_job.opponent_profile_id, expected_job.master_seed)
        ],
    )
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_postrun._load_pair_checkpoint",
        lambda path, *, binding, treatment, control: SimpleNamespace(
            scientifically_valid=True, checkpoint_hash=_hash(path.name)
        ),
    )
    profile_index = {profile: index for index, profile in enumerate(PROFILES)}
    seed_index = {seed: index for index, seed in enumerate(SEEDS)}
    finalize_calls: list[tuple[str, int]] = []

    def fake_finalize(*, treatment_job, control_job, **kwargs):  # type: ignore[no-untyped-def]
        assert treatment_job.opponent_profile_id == control_job.opponent_profile_id
        assert treatment_job.master_seed == control_job.master_seed
        finalize_calls.append((treatment_job.opponent_profile_id, treatment_job.master_seed))
        return _pair_execution(
            profile_index[treatment_job.opponent_profile_id], treatment_job.opponent_profile_id,
            seed_index[treatment_job.master_seed], treatment_job.master_seed,
        )

    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_postrun.finalize_adaptive_pilot_pair",
        fake_finalize,
    )
    all_diagnostics = _diagnostics()
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_postrun.learning_horizon_diagnostics",
        lambda record, *, profile_id, master_seed, horizons: tuple(
            item for item in all_diagnostics
            if item.profile_id == profile_id and item.master_seed == master_seed
        ),
    )
    ids = tuple(state.match_id for state in states)
    if extra_store_id:
        ids = (*ids, "foreign-match")
    fake_store = _FakeAuditStore(ids, records)
    monkeypatch.setattr(
        "pokerbench.experiments.adaptive_pilot_postrun.SQLiteBenchmarkStore",
        lambda path: fake_store,
    )
    database = tmp_path / "pilot.sqlite"
    database.write_bytes(b"sqlite-test")
    audit = _audit_completed_campaign(
        protocol=protocol,
        binding=binding,  # type: ignore[arg-type]
        database=database,
        work_root=tmp_path / "work",
    )
    return audit, finalize_calls


def test_campaign_audit_revalidates_exact_50_records_and_all_25_pairs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audit, finalize_calls = _exercise_campaign_audit(monkeypatch, tmp_path)
    assert audit.completed_jobs == 50 and audit.failed_jobs == 0
    assert audit.valid_pairs == 25 and audit.invalid_pairs == 0
    assert audit.recorded_engine_hands == 18_000
    assert audit.replayed_learning_transitions == 9_000
    assert len(finalize_calls) == 25 and len(set(finalize_calls)) == 25
    assert len(audit.learning_diagnostics) == 210


def test_campaign_audit_rejects_any_foreign_sqlite_match_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(AdaptivePilotPostRunError, match="SQLite MatchRecord identities"):
        _exercise_campaign_audit(monkeypatch, tmp_path, extra_store_id=True)
