from __future__ import annotations

import json
from pathlib import Path

from pokerbench.experiments.frozen_protocol import canonical_file_bytes


ROOT = Path(__file__).resolve().parents[2]
FROZEN_JSON_PATHS = (
    ROOT / "configs" / "experiments" / "phase1-pilot-v1.json",
    ROOT / "configs" / "populations" / "baseline-synthetic-v1.json",
    ROOT / "configs" / "populations" / "balanced-v1.json",
)


def test_frozen_json_paths_are_forced_to_lf_by_gitattributes() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()

    assert "configs/experiments/*.json text eol=lf" in attributes
    assert "configs/populations/*.json text eol=lf" in attributes


def test_all_frozen_phase1_json_files_are_canonical_bytes() -> None:
    for path in FROZEN_JSON_PATHS:
        raw = path.read_bytes()
        assert raw == canonical_file_bytes(json.loads(raw)), path
