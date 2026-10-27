# Public Plot CLI v1

## Status

This document freezes the first public plotting command required by the software deliverables.

```text
public_plot_cli_version = public-plot-cli-v1
plot_bundle_version     = battle-forest-plot-v1
source_battle_version   = public-battle-cli-v1
```

This specification is normative for `poker plot` v1.

## 1. Purpose

`poker plot` is a read-only presentation layer over already-computed scientific battle reports. It MUST NOT execute matches, replay hands for statistical recomputation, alter persisted benchmark data, tune policies, replace failed seeds, or recompute the paired estimator or confidence interval.

The v1 source of truth is one or more complete, hash-verified JSON reports produced by `public-battle-cli-v1`.

## 2. Command

One report:

```bash
poker plot \
  --report data/reports/random-vs-tag.json \
  --output data/plots/random-vs-tag
```

Multiple reports may be supplied by repeating `--report`:

```bash
poker plot \
  --report data/reports/random-vs-tag.json \
  --report data/reports/tag-vs-lag.json \
  --output data/plots/comparison
```

The confidence interval shown by default is the already-computed percentile bootstrap interval. The already-computed normal interval may be selected explicitly:

```bash
poker plot \
  --report data/reports/tag-vs-lag.json \
  --output data/plots/tag-vs-lag-normal \
  --ci normal
```

Allowed `--ci` values are exactly:

```text
bootstrap
normal
```

The output path is a directory and MUST NOT already exist. Existing scientific evidence is never silently overwritten.

## 3. Source validation

Every source file MUST contain the canonical battle-report envelope:

```text
report
report_hash
```

The command recomputes SHA-256 over the canonical JSON representation of `report` and requires exact equality with `report_hash`.

Each selected report MUST additionally satisfy all of the following:

- `battle_cli_version == public-battle-cli-v1`;
- `complete == true`;
- `failed_runs == 0`;
- `statistical_unit == paired_deal`;
- a statistical `evaluation` is present;
- the evaluation also uses `paired_deal`;
- the evaluation independent-unit and analyzed-hand counts agree with the report-level counts;
- the evaluation failure count agrees with the report-level failure count;
- the report and evaluation confidence levels are both exactly `0.95`;
- the selected mean and CI endpoints are finite numbers and the CI lower endpoint is not greater than the upper endpoint.

Incomplete or integrity-failing battle reports are not plotted. Duplicate source evidence is detected by `report_hash` and rejected even if it is supplied through different file paths.

## 4. Scientific quantity displayed

Each row is one complete battle report and displays the already-computed:

```text
mean_difference_bb100
```

as the point estimate.

For `--ci bootstrap`, the whisker uses:

```text
bootstrap_ci_low_bb100
bootstrap_ci_high_bb100
```

For `--ci normal`, the whisker uses:

```text
normal_ci_low_bb100
normal_ci_high_bb100
```

The horizontal axis is labelled:

```text
Paired EV difference (BB/100)
```

A vertical zero-reference line is always shown. The x-axis bounds always include zero, every plotted point estimate, and every selected CI endpoint. Plotting is therefore a presentation transformation only; no new estimator is introduced.

## 5. Deterministic ordering and SVG

Rows are ordered lexically by:

```text
(bot_a, bot_b, report_hash)
```

so input argument order does not change the output bytes.

`public-plot-cli-v1` uses a dependency-free deterministic SVG forest plot. It does not add Matplotlib or another plotting dependency to the frozen Poetry environment.

The SVG contains, for every source report:

- bot A and bot B labels;
- valid independent-unit count;
- mean paired difference in BB/100;
- selected 95% CI in BB/100;
- horizontal CI whisker;
- point estimate marker;
- zero-reference line.

The plotting geometry, tick positions, number formatting, and styling are deterministic functions of the validated point payloads. No timestamp, machine path, random identifier, or output destination is embedded in the SVG.

## 6. Bundle contents and integrity

A successful output directory contains exactly:

```text
plot.svg
manifest.json
```

`manifest.json` records at minimum:

```text
public_plot_cli_version
plot_bundle_version
source_battle_version
ci_kind
confidence_level
report_count
source_report_hashes
points
files -> SHA-256 of plot.svg
bundle_hash
```

The point provenance includes the source battle identity, bot versions, selected estimate and interval, independent-unit and analyzed-hand counts, game-configuration hash, Gate-6 report hash when applicable, and evaluation version.

`bundle_hash` is SHA-256 over canonical JSON of the manifest payload before the `bundle_hash` field is inserted. The manifest does not recursively hash itself.

Two invocations over the same source battle evidence with the same CI selection MUST produce byte-identical `plot.svg` and `manifest.json`, independent of source-path location, output-path location, and input argument order.

## 7. Failure and atomicity

All source reports are loaded and validated before the requested output directory is committed. The bundle is built in a temporary sibling directory and atomically renamed only after `plot.svg`, its SHA-256, and `manifest.json` are complete.

A failed plot command MUST NOT leave the requested output directory as partially valid evidence.

## 8. Pareto boundary

The project later requires a Pareto visualization of commensurable NLHE objectives such as mean population EV and `R10`. Those quantities are not produced by `public-battle-cli-v1`. `public-plot-cli-v1` therefore MUST NOT invent or infer a Pareto point from pairwise battle reports.

Pareto plotting becomes a later versioned plot mode only after the required aggregated population and robustness metrics are persisted under a frozen scientific contract.
