from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from .modeling.baseline import VisionBaselineConfig, build_vision_baseline


def find_checkpoint(run_dir: Path) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    candidates = [
        checkpoints_dir / "best.ckpt",
        checkpoints_dir / "last.ckpt",
    ]
    if checkpoints_dir.exists():
        candidates.extend(sorted(checkpoints_dir.glob("*.ckpt")))
    candidates.extend(
        [
            run_dir / "best_model.pt",
            run_dir / "baseline_model.pt",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No checkpoint found in {run_dir} (looked for checkpoints/*.ckpt, best_model.pt, baseline_model.pt)"
    )


def export_model(
    run_dir: Path,
    output_path: Path,
    *,
    export_format: str,
    device: str,
    opset: int,
) -> None:
    checkpoint_path = find_checkpoint(run_dir)
    config_path_candidates = [
        run_dir / "config.json",
        run_dir / "experiment_config.json",
    ]
    config: dict = {}
    for config_path in config_path_candidates:
        if config_path.exists():
            with config_path.open("r") as f:
                config = json.load(f)
            break
    if not config:
        raise FileNotFoundError(
            "Could not find config.json or experiment_config.json in the run directory"
        )

    model = build_vision_baseline(VisionBaselineConfig())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {
            key.replace("model.", ""): value for key, value in checkpoint["state_dict"].items()
        }
    else:
        state_dict = checkpoint
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing_keys = sorted(set(state_dict) - set(compatible_state))
    if missing_keys:
        print(
            "Skipping incompatible parameters:",
            ", ".join(missing_keys),
        )
    model.load_state_dict(compatible_state, strict=False)
    model.eval()

    device_obj = torch.device(device)
    model.to(device_obj)

    image_size = config.get("augmentation", {}).get("resize", [512, 648])
    if isinstance(image_size, Sequence) and len(image_size) == 2:
        height, width = image_size
    else:
        height, width = 512, 648

    dummy_input = torch.randn(1, 3, height, width, device=device_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if export_format == "torchscript":
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(output_path))
    elif export_format == "onnx":
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["image"],
            output_names=["class_logits", "mask_logits"],
            dynamic_axes={"image": {0: "batch"}, "class_logits": {0: "batch"}, "mask_logits": {0: "batch"}},
            opset_version=opset,
        )
    else:
        raise ValueError(f"Unsupported export format: {export_format}")

    normalization = {
        "image_scale": [0.0, 1.0],
        "input_shape": [height, width],
    }
    with (output_path.parent / "input_normalization.json").open("w") as f:
        json.dump(normalization, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export forgery baseline model")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_model(
        run_dir=args.run_dir,
        output_path=args.output,
        export_format=args.format,
        device=args.device,
        opset=args.opset,
    )
    print(f"Model exported to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
