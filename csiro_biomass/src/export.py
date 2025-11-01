from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from .config.experiment import ExperimentConfig, FusionConfig, OptimizerConfig, CurriculumConfig, SnapshotConfig, BackboneConfig
from .modeling.baseline import ModelSpec, build_model
from .modeling.baseline import get_normalization_stats


class InferenceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, metadata_dim: int) -> None:
        super().__init__()
        self.model = model
        self.metadata_dim = metadata_dim

    def forward(self, images: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.metadata_dim == 0:
            mean, _ = self.model(images, None)
        else:
            if metadata is None:
                raise ValueError("Metadata tensor expected but not provided.")
            mean, _ = self.model(images, metadata)
        return mean


def load_config(config_path: Path) -> dict:
    with config_path.open("r") as f:
        return json.load(f)


def build_model_from_config(config_dict: dict, metadata_dim: int, metadata_entropy: float) -> torch.nn.Module:
    fusion_cfg = FusionConfig(**config_dict.get("fusion", {}))
    backbone_cfg = BackboneConfig(**config_dict.get("backbone", {}))
    model_spec = ModelSpec(
        model_name=backbone_cfg.name,
        pretrained=backbone_cfg.pretrained,
        dropout=backbone_cfg.dropout,
        num_outputs=config_dict.get("num_targets", 4),
        metadata_dim=metadata_dim,
        metadata_entropy=max(metadata_entropy, 1e-3),
        tabular_hidden_dims=tuple(fusion_cfg.tabular_hidden_dims),
        fusion_hidden_dims=tuple(fusion_cfg.fusion_hidden_dims),
        fusion_dropout=fusion_cfg.fusion_dropout,
        use_layernorm=fusion_cfg.use_layernorm,
    )
    return build_model(model_spec, metadata_dim=metadata_dim)


def export_model(
    run_dir: Path,
    output_path: Path,
    *,
    export_format: str,
    opset: int,
    device: str,
) -> None:
    config_dict = load_config(run_dir / "config.json")
    metadata_info_path = run_dir / "metadata_info.json"
    metadata_columns: Sequence[str] = []
    metadata_mean = None
    metadata_std = None
    if metadata_info_path.exists():
        with metadata_info_path.open("r") as f:
            metadata_info = json.load(f)
        metadata_columns = metadata_info.get("columns", [])
        if metadata_info.get("mean") is not None:
            metadata_mean = torch.tensor(metadata_info["mean"], dtype=torch.float32)
        if metadata_info.get("std") is not None:
            metadata_std = torch.tensor(metadata_info["std"], dtype=torch.float32)
    metadata_dim = len(metadata_columns)
    metadata_entropy = 0.0
    if metadata_std is not None:
        # simple heuristic: higher variance -> higher entropy approx
        metadata_entropy = float(torch.log(metadata_std + 1e-6).abs().mean().item())

    model = build_model_from_config(config_dict, metadata_dim, metadata_entropy)
    state_dict = torch.load(run_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    wrapper = InferenceWrapper(model, metadata_dim)
    wrapper.eval()
    device_obj = torch.device(device)
    wrapper.to(device_obj)

    image_size = config_dict.get("augmentation", {}).get("image_size", 352)
    dummy_image = torch.randn(1, 3, image_size, image_size, device=device_obj)
    if metadata_dim > 0:
        dummy_metadata = torch.zeros(1, metadata_dim, device=device_obj)
        if metadata_mean is not None and metadata_std is not None:
            dummy_metadata = dummy_metadata * metadata_std.to(device_obj) + metadata_mean.to(device_obj)
    else:
        dummy_metadata = None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "torchscript":
        traced = torch.jit.trace(wrapper, (dummy_image, dummy_metadata) if metadata_dim > 0 else (dummy_image,))
        traced.save(str(output_path))
    elif export_format == "onnx":
        onnx_inputs = (dummy_image, dummy_metadata) if metadata_dim > 0 else (dummy_image,)
        input_names = ["images"]
        if metadata_dim > 0:
            input_names.append("metadata")
        torch.onnx.export(
            wrapper,
            onnx_inputs,
            str(output_path),
            input_names=input_names,
            output_names=["mean"],
            dynamic_axes={
                name: {0: "batch"} for name in input_names
            },
            opset_version=opset,
        )
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    mean, std = get_normalization_stats()
    with (output_path.parent / "input_normalization.json").open("w") as f:
        json.dump(
            {
                "image_mean": mean,
                "image_std": std,
                "metadata_columns": list(metadata_columns),
                "metadata_mean": metadata_mean.tolist() if metadata_mean is not None else None,
                "metadata_std": metadata_std.tolist() if metadata_std is not None else None,
            },
            f,
            indent=2,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CSIRO biomass model to TorchScript or ONNX")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory containing config.json and best_model.pt")
    parser.add_argument("--output", type=Path, required=True, help="Destination file for exported model")
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    export_model(
        run_dir=args.run_dir,
        output_path=args.output,
        export_format=args.format,
        opset=args.opset,
        device=args.device,
    )
    print(f"Exported model to {args.output} in {args.format} format.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
