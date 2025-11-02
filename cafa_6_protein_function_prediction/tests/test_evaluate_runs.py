"""Tests for the evaluation aggregation CLI."""

import json
from pathlib import Path

from src.utils.evaluate_runs import _aggregate, _load_run_metrics, main


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_load_run_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _write_json(run_dir / "summary.json", {"fmax": 0.5, "epochs": 3})
    _write_json(run_dir / "cafa_metrics.json", {"coverage": 0.8})

    metrics = _load_run_metrics(run_dir)
    assert metrics["fmax"] == 0.5
    assert metrics["coverage"] == 0.8
    assert "__run_path__" in metrics


def test_aggregate_cli(tmp_path, capsys):
    run_a = tmp_path / "run_a"
    _write_json(run_a / "summary.json", {"fmax": 0.5, "coverage": 0.9})
    run_b = tmp_path / "run_b"
    _write_json(run_b / "summary.json", {"fmax": 0.7, "coverage": 0.95})

    metrics_file = tmp_path / "ensemble.metrics.json"
    _write_json(metrics_file, {"fmax": 0.8, "coverage": 1.0})

    output = tmp_path / "agg.json"
    args = [
        "--run-dir",
        str(run_a),
        "--run-dir",
        str(run_b),
        "--metrics-file",
        str(metrics_file),
        "--label",
        "ensemble",
        "--json-output",
        str(output),
        "--log-level",
        "WARNING",
    ]
    exit_code = main(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "fmax" in captured
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "aggregates" in payload
    assert abs(payload["aggregates"]["fmax"]["mean"] - 0.6666667) < 1e-6
