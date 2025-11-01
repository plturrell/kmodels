from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .config.experiment import ModelConfig
from .modeling import build_baseline_model


class AnalyticsInferenceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(features)


def export_model(
    run_dir: Path,
    output_path: Path,
    *,
    export_format: str,
    device: str,
    opset: int,
) -> None:
    config_path = run_dir / "config.json"
    metadata_path = run_dir / "feature_metadata.json"
    state_path = run_dir / "best_model.pt"

    if not config_path.exists() or not metadata_path.exists() or not state_path.exists():
        raise FileNotFoundError("Run directory must contain config.json, feature_metadata.json, and best_model.pt")

    with config_path.open("r") as f:
        config_dict = json.load(f)
    with metadata_path.open("r") as f:
        metadata = json.load(f)

    feature_columns: Sequence[str] = metadata.get("feature_columns", [])
    target_columns: Sequence[str] = metadata.get("target_columns", [])
    input_dim = len(feature_columns)
    output_dim = len(target_columns)

    model_cfg = ModelConfig(
        architecture=config_dict.get("model", {}).get("architecture", "mlp"),
        hidden_dims=config_dict.get("model", {}).get("hidden_dims", [512, 256]),
        dropout=config_dict.get("model", {}).get("dropout", 0.1),
        n_heads=config_dict.get("model", {}).get("n_heads", 4),
        sequence_length=config_dict.get("model", {}).get("sequence_length", 64),
        use_metadata=False,
    )

    model = build_baseline_model(
        input_dim=input_dim,
        output_dim=output_dim,
        model_cfg=model_cfg,
    )
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    wrapper = AnalyticsInferenceWrapper(model).to(device)
    dummy_features = torch.randn(1, input_dim, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if export_format == "torchscript":
        traced = torch.jit.trace(wrapper, dummy_features)
        traced.save(str(output_path))
    elif export_format == "onnx":
        torch.onnx.export(
            wrapper,
            dummy_features,
            str(output_path),
            input_names=["features"],
            output_names=["prediction"],
            dynamic_axes={"features": {0: "batch"}, "prediction": {0: "batch"}},
            opset_version=opset,
        )
    else:
        raise ValueError(f"Unknown export format: {export_format}")

    normalization_info = {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "feature_mean": metadata.get("feature_mean"),
        "feature_std": metadata.get("feature_std"),
    }
    with (output_path.parent / "input_normalization.json").open("w") as f:
        json.dump(normalization_info, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export analytics baseline model")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args()
    export_model(
        run_dir=args.run_dir,
        output_path=args.output,
        export_format=args.format,
        device=args.device,
        opset=args.opset,
    )
    print(f"Exported model to {args.output} in {args.format} format.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
