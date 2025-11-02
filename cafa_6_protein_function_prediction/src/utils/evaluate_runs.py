"""Aggregate metrics across Lightning runs and ensemble outputs."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

LOGGER = logging.getLogger(__name__)


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Expected JSON file at {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_run_metrics(run_dir: Path) -> Dict[str, float]:
    """Load metrics from a Lightning run directory."""
    summary_path = run_dir / "summary.json"
    cafa_path = run_dir / "cafa_metrics.json"

    metrics: Dict[str, float] = {}
    if summary_path.exists():
        metrics.update(
            {
                key: float(value)
                for key, value in _load_json(summary_path).items()
                if isinstance(value, (int, float))
            }
        )
    if cafa_path.exists():
        summary = _load_json(cafa_path)
        metrics.update(
            {
                key: float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float))
            }
        )
    if not metrics:
        raise ValueError(f"No numeric metrics found under {run_dir}")
    metrics["__run_path__"] = str(run_dir)
    return metrics


def _load_metrics_file(path: Path, label: Optional[str] = None) -> Dict[str, float]:
    metrics = {
        key: float(value)
        for key, value in _load_json(path).items()
        if isinstance(value, (int, float))
    }
    if not metrics:
        raise ValueError(f"No numeric values found in {path}")
    metrics["__label__"] = label or path.stem
    metrics["__metrics_path__"] = str(path)
    return metrics


def _aggregate(rows: Iterable[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    collected: Dict[str, List[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("__"):
                continue
            collected.setdefault(key, []).append(value)

    aggregates: Dict[str, Dict[str, float]] = {}
    for key, values in collected.items():
        aggregates[key] = {
            "count": float(len(values)),
            "mean": statistics.fmean(values) if values else float("nan"),
            "median": float(statistics.median(values)) if values else float("nan"),
            "min": min(values),
            "max": max(values),
        }
    return aggregates


def _format_table(rows: List[Dict[str, float]], aggregates: Dict[str, Dict[str, float]]) -> str:
    labels: List[str] = []
    for row in rows:
        label = row.get("__label__")
        source = row.get("__run_path__") or row.get("__metrics_path__")
        if label is None and isinstance(source, str):
            label = Path(source).stem
        labels.append(label or "metrics")

    header = ["metric"] + labels + [
        "mean",
        "median",
        "min",
        "max",
    ]

    metric_names = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if not key.startswith("__")
        }
    )

    lines = ["\t".join(header)]
    for key in metric_names:
        line = [key]
        for row in rows:
            value = row.get(key)
            line.append(f"{value:.4f}" if value is not None else "-")
        agg = aggregates.get(key)
        if agg:
            line.extend(
                [
                    f"{agg['mean']:.4f}",
                    f"{agg['median']:.4f}",
                    f"{agg['min']:.4f}",
                    f"{agg['max']:.4f}",
                ]
            )
        else:
            line.extend(["-"] * 4)
        lines.append("\t".join(line))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate metrics across Lightning runs and ensemble outputs.")
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=[],
        help="Path to a Lightning run directory containing summary.json/cafa_metrics.json. Repeatable.",
    )
    parser.add_argument(
        "--metrics-file",
        action="append",
        type=Path,
        default=[],
        help="Standalone metrics JSON (e.g., from ensemble_cli). Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Optional label for the last provided --metrics-file.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to export aggregated stats as JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    rows: List[Dict[str, float]] = []

    for run_path in args.run_dir:
        run_dir = run_path.resolve()
        LOGGER.info("Loading run metrics from %s", run_dir)
        row = _load_run_metrics(run_dir)
        row["__label__"] = run_dir.name
        rows.append(row)

    label_queue = list(args.label or [])
    for metrics_path in args.metrics_file:
        path = metrics_path.resolve()
        label = label_queue.pop(0) if label_queue else None
        LOGGER.info("Loading metrics from %s", path)
        row = _load_metrics_file(path, label=label or path.stem)
        rows.append(row)

    if not rows:
        parser.error("Provide at least one --run-dir or --metrics-file.")

    aggregates = _aggregate(rows)
    table = _format_table(rows, aggregates)
    print(table)

    if args.json_output:
        payload = {
            "rows": rows,
            "aggregates": aggregates,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Saved aggregated metrics to %s", args.json_output)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
