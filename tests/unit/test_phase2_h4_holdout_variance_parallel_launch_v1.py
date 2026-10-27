from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from pokerbench.experiments.h4_holdout_variance_execution_contract import (
    H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH,
)
from pokerbench.experiments.h4_holdout_variance_parallel_launch import (
    H4HoldoutVarianceParallelLaunchError,
    assert_h4_holdout_variance_parallel_launch_ready,
    plan_h4_holdout_variance_parallel_launch,
    status_h4_holdout_variance_parallel_launch,
)
from pokerbench.persistence.sqlite import SQLiteBenchmarkStore


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-pilot-v1.json"
CONTRACT_PATH = ROOT / "configs" / "experiments" / "phase2-h4-holdout-variance-execution-v1.json"
POPULATION_PATH = ROOT / "configs" / "populations" / "phase2-h4-holdout-population-v1.json"
ACCEPTANCE_PATH = ROOT / "data" / "gate6" / "h4-holdout-acceptance-v1.json"
SNAPSHOT_ROOT = ROOT / "data" / "gate6" / "snapshots"
GATE6_PATH = ROOT / "data" / "gate6" / "gate6-report.json"
REFERENCE_PATH = ROOT / "configs" / "experiments" / "preflop-range-strength-reference-v1.json"


def _repo(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "docs" / "specs").mkdir(parents=True)
    (repo / "src" / "marker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "configs" / "marker.json").write_text("{}\n", encoding="utf-8")
    (repo / "docs" / "specs" / "marker.md").write_text("# marker\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[tool.poetry]\nname='spec38-launch-test'\n", encoding="utf-8")
    lock = repo / "poetry.lock"
    lock.write_text("# frozen lock\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "clean"], cwd=repo, check=True)
    database = repo / "data" / "phase2" / "h4-holdout-variance" / "variance-pilot.sqlite"
    work_root = repo / "data" / "phase2" / "h4-holdout-variance" / "work"
    return repo, lock, database, work_root


def _kwargs(tmp_path: Path) -> dict[str, object]:
    repo, lock, database, work_root = _repo(tmp_path)
    return {
        "protocol_path": PROTOCOL_PATH,
        "execution_contract_path": CONTRACT_PATH,
        "population_path": POPULATION_PATH,
        "acceptance_path": ACCEPTANCE_PATH,
        "snapshot_root": SNAPSHOT_ROOT,
        "gate6_report_path": GATE6_PATH,
        "reference_path": REFERENCE_PATH,
        "database": database,
        "work_root": work_root,
        "repo_root": repo,
        "poetry_lock_path": lock,
    }


def _bytes(root: Path, database: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(database) + suffix)
        if path.exists():
            result[f"database{suffix}"] = path.read_bytes()
    return result


def test_plan_materializes_exact_100_job_virgin_campaign_and_zero_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    status = plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=16 * 1024**3, workers=4)

    assert status.virgin
    assert (status.planned_jobs, status.planned_pairs, status.planned_clusters, status.planned_engine_hands) == (
        100,
        50,
        10,
        100_000,
    )
    assert status.terminal_jobs == status.running == status.completed == status.failed == 0
    work_root = Path(kwargs["work_root"])
    assert len(tuple((work_root / "trajectory-configs").glob("*.json"))) == 100
    assert not tuple((work_root / "arm-snapshots").iterdir())
    assert not tuple((work_root / "pair-checkpoints").iterdir())
    assert not tuple((work_root / "cluster-checkpoints").iterdir())
    assert not SQLiteBenchmarkStore(Path(kwargs["database"])).list_match_ids()
    assert H4_HOLDOUT_VARIANCE_EXECUTION_CONTRACT_HASH.encode() in (work_root / "binding.json").read_bytes()


def test_plan_rejects_worker_override_before_creating_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    with pytest.raises(H4HoldoutVarianceParallelLaunchError, match="workers=4"):
        plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=3)
    assert not Path(kwargs["work_root"]).exists()
    assert not Path(kwargs["database"]).exists()


def test_plan_rejects_nonfrozen_persistence_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    kwargs["work_root"] = Path(kwargs["repo_root"]) / "data" / "phase2" / "wrong-root"
    with pytest.raises(H4HoldoutVarianceParallelLaunchError, match="persistence paths"):
        plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)


def test_plan_rejects_host_with_fewer_than_four_logical_cores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 2)
    kwargs = _kwargs(tmp_path)
    with pytest.raises(H4HoldoutVarianceParallelLaunchError, match="four logical CPU"):
        plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)


def test_status_is_read_only_and_does_not_recapture_live_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    planned = plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    before = _bytes(Path(kwargs["work_root"]), Path(kwargs["database"]))
    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_launch.capture_h4_holdout_variance_execution_binding",
        lambda **_: (_ for _ in ()).throw(AssertionError("status must not recapture environment")),
    )
    status = status_h4_holdout_variance_parallel_launch(
        protocol_path=PROTOCOL_PATH,
        execution_contract_path=CONTRACT_PATH,
        population_path=POPULATION_PATH,
        work_root=kwargs["work_root"],
    )
    assert status == planned
    assert _bytes(Path(kwargs["work_root"]), Path(kwargs["database"])) == before


def test_readiness_recaptures_exact_binding_and_is_byte_nonmutating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    planned = plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=12 * 1024**3, workers=4)
    before = _bytes(Path(kwargs["work_root"]), Path(kwargs["database"]))
    ready = assert_h4_holdout_variance_parallel_launch_ready(**kwargs)
    after = _bytes(Path(kwargs["work_root"]), Path(kwargs["database"]))

    assert ready.binding_hash == planned.binding_hash
    assert (ready.planned_jobs, ready.planned_pairs, ready.planned_clusters, ready.planned_engine_hands) == (
        100,
        50,
        10,
        100_000,
    )
    assert ready.terminal_jobs == ready.pair_count == ready.cluster_count == 0
    assert before == after


def test_readiness_fails_closed_on_scientific_git_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    (Path(kwargs["repo_root"]) / "src" / "drift.py").write_text("DRIFT = True\n", encoding="utf-8")
    with pytest.raises(H4HoldoutVarianceParallelLaunchError, match="recapture current Spec-38 binding"):
        assert_h4_holdout_variance_parallel_launch_ready(**kwargs)


def test_first_hand_readiness_rejects_nonvirgin_manifest_without_reconciliation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    manifest_path = Path(kwargs["work_root"]) / "run-manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["manifest"]["jobs"][0]["status"] = "running"
    canonical_manifest = json.dumps(
        document["manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    document["manifest_hash"] = hashlib.sha256(canonical_manifest).hexdigest()
    manifest_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    before = manifest_path.read_bytes()
    with pytest.raises(H4HoldoutVarianceParallelLaunchError, match="virgin plan"):
        assert_h4_holdout_variance_parallel_launch_ready(**kwargs)
    assert manifest_path.read_bytes() == before


def test_run_accepts_only_virgin_plan_and_delegates_to_parallel_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pokerbench.experiments.h4_holdout_variance_parallel_launch import (
        run_h4_holdout_variance_parallel_launch,
    )
    from pokerbench.experiments.h4_holdout_variance_run import H4HoldoutVarianceRunResult

    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    sentinel = H4HoldoutVarianceRunResult(
        binding_hash="a" * 64,
        terminal_jobs=100,
        completed_jobs=100,
        failed_jobs=0,
        pair_count=50,
        valid_pairs=50,
        invalid_pairs=0,
        cluster_count=10,
        valid_clusters=10,
        invalid_clusters=0,
        scientifically_complete=True,
    )
    calls: list[object] = []
    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_launch.execute_h4_holdout_variance_parallel_run",
        lambda **call: calls.append(call) or sentinel,
    )
    result = run_h4_holdout_variance_parallel_launch(**kwargs, resume=False)
    assert result == sentinel
    assert len(calls) == 1


def test_run_rejects_nonvirgin_campaign_and_directs_operator_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pokerbench.experiments.h4_holdout_variance_parallel_launch import (
        run_h4_holdout_variance_parallel_launch,
    )

    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    manifest_path = Path(kwargs["work_root"]) / "run-manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["manifest"]["jobs"][0]["status"] = "running"
    canonical_manifest = json.dumps(
        document["manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    document["manifest_hash"] = hashlib.sha256(canonical_manifest).hexdigest()
    manifest_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(H4HoldoutVarianceParallelLaunchError, match="use resume"):
        run_h4_holdout_variance_parallel_launch(**kwargs, resume=False)


def test_resume_accepts_same_bound_nonvirgin_campaign_without_first_hand_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pokerbench.experiments.h4_holdout_variance_parallel_launch import (
        run_h4_holdout_variance_parallel_launch,
    )
    from pokerbench.experiments.h4_holdout_variance_run import H4HoldoutVarianceRunResult

    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    manifest_path = Path(kwargs["work_root"]) / "run-manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["manifest"]["jobs"][0]["status"] = "running"
    canonical_manifest = json.dumps(
        document["manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    document["manifest_hash"] = hashlib.sha256(canonical_manifest).hexdigest()
    manifest_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_launch.assert_h4_holdout_variance_parallel_launch_ready",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("resume must not require virgin first-hand readiness")),
    )
    sentinel = H4HoldoutVarianceRunResult(
        binding_hash="a" * 64,
        terminal_jobs=100,
        completed_jobs=100,
        failed_jobs=0,
        pair_count=50,
        valid_pairs=50,
        invalid_pairs=0,
        cluster_count=10,
        valid_clusters=10,
        invalid_clusters=0,
        scientifically_complete=True,
    )
    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_launch.execute_h4_holdout_variance_parallel_run",
        lambda **call: sentinel,
    )
    assert run_h4_holdout_variance_parallel_launch(**kwargs, resume=True) == sentinel


def test_plan_and_readiness_cannot_invoke_parallel_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_launch.execute_h4_holdout_variance_parallel_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("plan/readiness must not invoke runtime")),
    )
    kwargs = _kwargs(tmp_path)
    planned = plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    ready = assert_h4_holdout_variance_parallel_launch_ready(**kwargs)
    assert planned.virgin
    assert planned.binding_hash == ready.binding_hash
    assert ready.terminal_jobs == ready.pair_count == ready.cluster_count == 0


def test_resume_fails_closed_on_scientific_git_drift_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pokerbench.experiments.h4_holdout_variance_parallel_launch import (
        run_h4_holdout_variance_parallel_launch,
    )

    monkeypatch.setattr("pokerbench.experiments.h4_holdout_variance_parallel_launch.os.cpu_count", lambda: 8)
    kwargs = _kwargs(tmp_path)
    plan_h4_holdout_variance_parallel_launch(**kwargs, ram_bytes=8 * 1024**3, workers=4)
    manifest_path = Path(kwargs["work_root"]) / "run-manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["manifest"]["jobs"][0]["status"] = "running"
    canonical_manifest = json.dumps(
        document["manifest"], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    document["manifest_hash"] = hashlib.sha256(canonical_manifest).hexdigest()
    manifest_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    (Path(kwargs["repo_root"]) / "src" / "drift.py").write_text("DRIFT = True\n", encoding="utf-8")
    monkeypatch.setattr(
        "pokerbench.experiments.h4_holdout_variance_parallel_launch.execute_h4_holdout_variance_parallel_run",
        lambda **_: (_ for _ in ()).throw(AssertionError("drift must fail before runtime")),
    )
    with pytest.raises(H4HoldoutVarianceParallelLaunchError, match="recapture current Spec-38 binding"):
        run_h4_holdout_variance_parallel_launch(**kwargs, resume=True)
