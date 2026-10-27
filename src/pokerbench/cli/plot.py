"""Deterministic SVG plots for complete public battle reports."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence


PUBLIC_PLOT_CLI_VERSION = "public-plot-cli-v1"
PLOT_BUNDLE_VERSION = "battle-forest-plot-v1"
SOURCE_BATTLE_VERSION = "public-battle-cli-v1"
PLOT_CONFIDENCE_LEVEL = 0.95

CIKind = Literal["bootstrap", "normal"]
_ALLOWED_CI_KINDS: tuple[CIKind, ...] = ("bootstrap", "normal")
_ALLOWED_BOTS = frozenset(
    {
        "random",
        "nit",
        "tag",
        "lag",
        "calling_station",
        "maniac",
    }
)

_SVG_WIDTH = 1100
_SVG_LEFT = 330
_SVG_RIGHT = 70
_SVG_TOP = 120
_SVG_ROW_HEIGHT = 72
_SVG_BOTTOM = 92
_SVG_TICK_COUNT = 5


@dataclass(frozen=True)
class BattlePlotPoint:
    """One integrity-checked point taken from a public battle report."""

    report_hash: str
    bot_a: str
    bot_b: str
    bot_a_version: str
    bot_b_version: str
    mean_difference_bb100: float
    ci_low_bb100: float
    ci_high_bb100: float
    ci_kind: CIKind
    valid_independent_units: int
    analyzed_engine_hands: int
    game_config_hash: str
    gate6_report_hash: str | None
    evaluation_version: str
    battle_cli_version: str = SOURCE_BATTLE_VERSION
    confidence_level: float = PLOT_CONFIDENCE_LEVEL

    def manifest_payload(self) -> dict[str, Any]:
        return {
            "report_hash": self.report_hash,
            "bot_a": self.bot_a,
            "bot_b": self.bot_b,
            "bot_a_version": self.bot_a_version,
            "bot_b_version": self.bot_b_version,
            "mean_difference_bb100": self.mean_difference_bb100,
            "ci_low_bb100": self.ci_low_bb100,
            "ci_high_bb100": self.ci_high_bb100,
            "ci_kind": self.ci_kind,
            "confidence_level": self.confidence_level,
            "valid_independent_units": self.valid_independent_units,
            "analyzed_engine_hands": self.analyzed_engine_hands,
            "game_config_hash": self.game_config_hash,
            "gate6_report_hash": self.gate6_report_hash,
            "evaluation_version": self.evaluation_version,
            "battle_cli_version": self.battle_cli_version,
        }


@dataclass(frozen=True)
class PublicPlotResult:
    """Paths and identity of one completed deterministic plot bundle."""

    output: Path
    svg_path: Path
    manifest_path: Path
    report_count: int
    ci_kind: CIKind
    bundle_hash: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _unique_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_unique_json(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object_pairs,
            parse_constant=lambda value: (_raise_nonfinite_json(value)),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON battle report: {path}") from exc


def _raise_nonfinite_json(value: str) -> Any:
    raise ValueError(f"Non-finite JSON number is not permitted: {value}")


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_ci_kind(ci_kind: str) -> CIKind:
    if ci_kind not in _ALLOWED_CI_KINDS:
        raise ValueError("ci_kind must be 'bootstrap' or 'normal'")
    return ci_kind  # type: ignore[return-value]


def load_battle_plot_point(
    path: str | Path,
    *,
    ci_kind: str = "bootstrap",
) -> BattlePlotPoint:
    """Load and validate one complete public-battle-v1 report for plotting."""
    selected_ci = _validate_ci_kind(ci_kind)
    source = Path(path)
    outer = _require_mapping(
        _load_unique_json(source),
        name="battle report envelope",
    )

    if set(outer) != {"report", "report_hash"}:
        raise ValueError("Battle report envelope must contain only report and report_hash")

    report = _require_mapping(outer["report"], name="report")
    report_hash = _require_string(outer["report_hash"], name="report_hash")
    if not _is_sha256(report_hash):
        raise ValueError("report_hash must be a lowercase SHA-256 value")

    try:
        calculated_hash = _sha256_bytes(
            _canonical_json(report).encode("utf-8")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Battle report cannot be canonically serialized") from exc
    if calculated_hash != report_hash:
        raise ValueError("Battle report hash mismatch")

    battle_version = _require_string(
        report.get("battle_cli_version"),
        name="battle_cli_version",
    )
    if battle_version != SOURCE_BATTLE_VERSION:
        raise ValueError(
            f"Unsupported battle_cli_version: {battle_version}"
        )

    complete = report.get("complete")
    if complete is not True:
        raise ValueError("Only complete public battle reports may be plotted")

    failed_runs = _require_int(
        report.get("failed_runs"),
        name="failed_runs",
    )
    if failed_runs != 0:
        raise ValueError("Complete plotted battle reports must have zero failed_runs")

    statistical_unit = _require_string(
        report.get("statistical_unit"),
        name="statistical_unit",
    )
    if statistical_unit != "paired_deal":
        raise ValueError("public-plot-cli-v1 requires paired_deal battle reports")

    confidence_level = _require_float(
        report.get("confidence_level"),
        name="confidence_level",
    )
    if confidence_level != PLOT_CONFIDENCE_LEVEL:
        raise ValueError("public-plot-cli-v1 requires confidence_level == 0.95")

    bot_a = _require_string(report.get("bot_a"), name="bot_a")
    bot_b = _require_string(report.get("bot_b"), name="bot_b")
    if bot_a not in _ALLOWED_BOTS or bot_b not in _ALLOWED_BOTS:
        raise ValueError("Battle report contains an unsupported public bot identity")

    bot_a_version = _require_string(
        report.get("bot_a_version"),
        name="bot_a_version",
    )
    bot_b_version = _require_string(
        report.get("bot_b_version"),
        name="bot_b_version",
    )
    valid_units = _require_int(
        report.get("valid_independent_units"),
        name="valid_independent_units",
        minimum=1,
    )
    analyzed_hands = _require_int(
        report.get("analyzed_engine_hands"),
        name="analyzed_engine_hands",
        minimum=1,
    )

    game_config_hash = _require_string(
        report.get("game_config_hash"),
        name="game_config_hash",
    )
    if not _is_sha256(game_config_hash):
        raise ValueError("game_config_hash must be a lowercase SHA-256 value")

    raw_gate6_hash = report.get("gate6_report_hash")
    gate6_hash: str | None
    if raw_gate6_hash is None:
        gate6_hash = None
    else:
        gate6_hash = _require_string(
            raw_gate6_hash,
            name="gate6_report_hash",
        )
        if not _is_sha256(gate6_hash):
            raise ValueError("gate6_report_hash must be a lowercase SHA-256 value")

    evaluation = _require_mapping(report.get("evaluation"), name="evaluation")
    evaluation_unit = _require_string(
        evaluation.get("statistical_unit"),
        name="evaluation.statistical_unit",
    )
    if evaluation_unit != "paired_deal":
        raise ValueError("Battle evaluation must use paired_deal units")

    evaluation_units = _require_int(
        evaluation.get("num_independent_units"),
        name="evaluation.num_independent_units",
        minimum=1,
    )
    if evaluation_units != valid_units:
        raise ValueError("Battle report/evaluation independent-unit counts disagree")

    evaluation_hands = _require_int(
        evaluation.get("num_analyzed_simulated_hands"),
        name="evaluation.num_analyzed_simulated_hands",
        minimum=1,
    )
    if evaluation_hands != analyzed_hands:
        raise ValueError("Battle report/evaluation analyzed-hand counts disagree")

    evaluation_failures = _require_int(
        evaluation.get("num_failed_runs"),
        name="evaluation.num_failed_runs",
    )
    if evaluation_failures != failed_runs:
        raise ValueError("Battle report/evaluation failed-run counts disagree")

    evaluation_confidence = _require_float(
        evaluation.get("confidence_level"),
        name="evaluation.confidence_level",
    )
    if evaluation_confidence != PLOT_CONFIDENCE_LEVEL:
        raise ValueError("Battle evaluation confidence level must equal 0.95")

    report_evaluation_version = _require_string(
        report.get("evaluation_version"),
        name="evaluation_version",
    )
    evaluation_version = _require_string(
        evaluation.get("evaluation_version"),
        name="evaluation.evaluation_version",
    )
    if evaluation_version != report_evaluation_version:
        raise ValueError("Battle report/evaluation version identifiers disagree")

    mean = _require_float(
        evaluation.get("mean_difference_bb100"),
        name="evaluation.mean_difference_bb100",
    )
    if selected_ci == "bootstrap":
        low_key = "bootstrap_ci_low_bb100"
        high_key = "bootstrap_ci_high_bb100"
    else:
        low_key = "normal_ci_low_bb100"
        high_key = "normal_ci_high_bb100"
    ci_low = _require_float(
        evaluation.get(low_key),
        name=f"evaluation.{low_key}",
    )
    ci_high = _require_float(
        evaluation.get(high_key),
        name=f"evaluation.{high_key}",
    )
    if ci_low > ci_high:
        raise ValueError("Selected CI lower endpoint exceeds upper endpoint")

    return BattlePlotPoint(
        report_hash=report_hash,
        bot_a=bot_a,
        bot_b=bot_b,
        bot_a_version=bot_a_version,
        bot_b_version=bot_b_version,
        mean_difference_bb100=mean,
        ci_low_bb100=ci_low,
        ci_high_bb100=ci_high,
        ci_kind=selected_ci,
        valid_independent_units=valid_units,
        analyzed_engine_hands=analyzed_hands,
        game_config_hash=game_config_hash,
        gate6_report_hash=gate6_hash,
        evaluation_version=evaluation_version,
        battle_cli_version=battle_version,
        confidence_level=confidence_level,
    )


def _format_value(value: float) -> str:
    normalized = 0.0 if abs(value) < 0.0005 else value
    return f"{normalized:.2f}"


def _coord(value: float) -> str:
    return f"{value:.3f}"


def _domain(points: Sequence[BattlePlotPoint]) -> tuple[float, float]:
    values = [0.0]
    for point in points:
        values.extend(
            (
                point.mean_difference_bb100,
                point.ci_low_bb100,
                point.ci_high_bb100,
            )
        )
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0.0:
        return (-1.0, 1.0)
    padding = max(span * 0.08, 0.25)
    return (low - padding, high + padding)


def render_battle_forest_svg(
    points: Sequence[BattlePlotPoint],
    *,
    ci_kind: str,
) -> str:
    """Render deterministic SVG from already-validated battle points."""
    selected_ci = _validate_ci_kind(ci_kind)
    if not points:
        raise ValueError("At least one battle point is required")
    if any(point.ci_kind != selected_ci for point in points):
        raise ValueError("All battle points must use the selected CI kind")

    ordered = tuple(
        sorted(
            points,
            key=lambda point: (
                point.bot_a,
                point.bot_b,
                point.report_hash,
            ),
        )
    )
    domain_low, domain_high = _domain(ordered)
    plot_width = _SVG_WIDTH - _SVG_LEFT - _SVG_RIGHT
    height = _SVG_TOP + len(ordered) * _SVG_ROW_HEIGHT + _SVG_BOTTOM
    axis_y = _SVG_TOP + len(ordered) * _SVG_ROW_HEIGHT - 16

    def x_coordinate(value: float) -> float:
        return _SVG_LEFT + (
            (value - domain_low)
            / (domain_high - domain_low)
            * plot_width
        )

    title_ci = "bootstrap" if selected_ci == "bootstrap" else "normal"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_WIDTH}" '
            f'height="{height}" viewBox="0 0 {_SVG_WIDTH} {height}" '
            'role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">PokerBench paired EV comparison</title>',
        (
            '<desc id="desc">Complete public-battle-cli-v1 reports shown as '
            f'paired EV differences with 95% {title_ci} confidence intervals.</desc>'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#111827">',
        '<text x="24" y="38" font-size="24" font-weight="700">PokerBench paired EV comparison</text>',
        (
            f'<text x="24" y="64" font-size="14" fill="#4b5563">'
            f'Point estimate with 95% {title_ci} CI (BB/100)</text>'
        ),
    ]

    for index in range(_SVG_TICK_COUNT):
        fraction = index / (_SVG_TICK_COUNT - 1)
        tick_value = domain_low + fraction * (domain_high - domain_low)
        tick_x = x_coordinate(tick_value)
        lines.extend(
            [
                (
                    f'<line x1="{_coord(tick_x)}" y1="86" '
                    f'x2="{_coord(tick_x)}" y2="{axis_y}" '
                    'stroke="#e5e7eb" stroke-width="1"/>'
                ),
                (
                    f'<text x="{_coord(tick_x)}" y="{axis_y + 28}" '
                    'font-size="12" text-anchor="middle" fill="#6b7280">'
                    f'{html.escape(_format_value(tick_value))}</text>'
                ),
            ]
        )

    zero_x = x_coordinate(0.0)
    lines.append(
        f'<line x1="{_coord(zero_x)}" y1="82" x2="{_coord(zero_x)}" '
        f'y2="{axis_y}" stroke="#9ca3af" stroke-width="1.5" '
        'stroke-dasharray="5 4"/>'
    )

    for row_index, point in enumerate(ordered):
        row_y = _SVG_TOP + row_index * _SVG_ROW_HEIGHT
        low_x = x_coordinate(point.ci_low_bb100)
        high_x = x_coordinate(point.ci_high_bb100)
        mean_x = x_coordinate(point.mean_difference_bb100)
        label = html.escape(f"{point.bot_a} vs {point.bot_b}")
        detail = html.escape(
            f"mean {_format_value(point.mean_difference_bb100)}; "
            f"95% CI [{_format_value(point.ci_low_bb100)}, "
            f"{_format_value(point.ci_high_bb100)}]; "
            f"n={point.valid_independent_units}"
        )
        lines.extend(
            [
                (
                    f'<text x="24" y="{row_y - 6}" font-size="15" '
                    f'font-weight="600">{label}</text>'
                ),
                (
                    f'<text x="24" y="{row_y + 15}" font-size="12" '
                    f'fill="#6b7280">{detail}</text>'
                ),
                (
                    f'<line x1="{_coord(low_x)}" y1="{row_y}" '
                    f'x2="{_coord(high_x)}" y2="{row_y}" '
                    'stroke="#4b5563" stroke-width="3"/>'
                ),
                (
                    f'<line x1="{_coord(low_x)}" y1="{row_y - 7}" '
                    f'x2="{_coord(low_x)}" y2="{row_y + 7}" '
                    'stroke="#4b5563" stroke-width="2"/>'
                ),
                (
                    f'<line x1="{_coord(high_x)}" y1="{row_y - 7}" '
                    f'x2="{_coord(high_x)}" y2="{row_y + 7}" '
                    'stroke="#4b5563" stroke-width="2"/>'
                ),
                (
                    f'<circle cx="{_coord(mean_x)}" cy="{row_y}" r="6" '
                    'fill="#111827"/>'
                ),
            ]
        )

    lines.extend(
        [
            (
                f'<line x1="{_SVG_LEFT}" y1="{axis_y}" '
                f'x2="{_SVG_WIDTH - _SVG_RIGHT}" y2="{axis_y}" '
                'stroke="#374151" stroke-width="1.5"/>'
            ),
            (
                f'<text x="{_SVG_LEFT + plot_width / 2:.3f}" '
                f'y="{axis_y + 58}" font-size="14" text-anchor="middle" '
                'font-weight="600">Paired EV difference (BB/100)</text>'
            ),
            (
                f'<text x="{_SVG_WIDTH - 24}" y="{height - 18}" '
                'font-size="11" text-anchor="end" fill="#9ca3af">'
                f'{PUBLIC_PLOT_CLI_VERSION}</text>'
            ),
            '</g>',
            '</svg>',
            '',
        ]
    )
    return "\n".join(lines)


def execute_public_plot(
    *,
    report_paths: Sequence[str | Path],
    output: str | Path,
    ci_kind: str = "bootstrap",
) -> PublicPlotResult:
    """Create one atomic deterministic SVG plot bundle."""
    if isinstance(report_paths, (str, bytes, Path)):
        raise TypeError("report_paths must be a sequence of paths")
    paths = tuple(Path(path) for path in report_paths)
    if not paths:
        raise ValueError("poker plot requires at least one --report")

    selected_ci = _validate_ci_kind(ci_kind)
    destination = Path(output)
    if destination.exists():
        raise ValueError(f"Plot output already exists: {destination}")

    points = tuple(
        load_battle_plot_point(path, ci_kind=selected_ci)
        for path in paths
    )
    report_hashes = [point.report_hash for point in points]
    if len(set(report_hashes)) != len(report_hashes):
        raise ValueError("Duplicate source battle report evidence is not permitted")

    ordered = tuple(
        sorted(
            points,
            key=lambda point: (
                point.bot_a,
                point.bot_b,
                point.report_hash,
            ),
        )
    )
    svg = render_battle_forest_svg(ordered, ci_kind=selected_ci)
    svg_bytes = svg.encode("utf-8")
    svg_hash = _sha256_bytes(svg_bytes)

    manifest_payload: dict[str, Any] = {
        "public_plot_cli_version": PUBLIC_PLOT_CLI_VERSION,
        "plot_bundle_version": PLOT_BUNDLE_VERSION,
        "source_battle_version": SOURCE_BATTLE_VERSION,
        "ci_kind": selected_ci,
        "confidence_level": PLOT_CONFIDENCE_LEVEL,
        "report_count": len(ordered),
        "source_report_hashes": [point.report_hash for point in ordered],
        "points": [point.manifest_payload() for point in ordered],
        "files": {"plot.svg": svg_hash},
    }
    bundle_hash = _sha256_bytes(
        _canonical_json(manifest_payload).encode("utf-8")
    )
    manifest = {
        **manifest_payload,
        "bundle_hash": bundle_hash,
    }
    manifest_bytes = (
        _canonical_json(manifest) + "\n"
    ).encode("utf-8")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=str(destination.parent),
        )
    )
    committed = False
    try:
        _write_bytes_fsync(temporary / "plot.svg", svg_bytes)
        _write_bytes_fsync(temporary / "manifest.json", manifest_bytes)
        if destination.exists():
            raise ValueError(f"Plot output already exists: {destination}")
        os.replace(temporary, destination)
        committed = True
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)

    return PublicPlotResult(
        output=destination,
        svg_path=destination / "plot.svg",
        manifest_path=destination / "manifest.json",
        report_count=len(ordered),
        ci_kind=selected_ci,
        bundle_hash=bundle_hash,
    )


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def render_public_plot_text(result: PublicPlotResult) -> str:
    return "\n".join(
        [
            "PokerBench plot complete",
            f"  reports: {result.report_count}",
            f"  CI: {result.ci_kind}",
            f"  SVG: {result.svg_path}",
            f"  manifest: {result.manifest_path}",
            f"  bundle hash: {result.bundle_hash}",
        ]
    )
