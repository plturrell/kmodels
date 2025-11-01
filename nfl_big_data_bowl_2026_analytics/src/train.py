"""Training entry point for the NFL Big Data Bowl 2026 analytics workspace."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import GroupKFold, KFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

from .config.experiment import (
    DatasetConfig,
    FeatureConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from .data import (
    AnalyticsDataset,
    compute_feature_stats,
    create_dataloader,
    extract_numeric_features,
    load_training_dataframe,
    split_train_validation,
)
from .modeling import build_baseline_model
from .features.player_kinematics import add_player_kinematics, add_distance_to_ball
from .features.relational import add_relational_features
from .training import AnalyticsLightningModule
from .utils.metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _determine_device(device_request: str) -> Tuple[str, torch.device]:
    request = device_request.lower()
    if request.startswith("cuda") and torch.cuda.is_available():
        return "gpu", torch.device("cuda")
    if request.startswith("mps"):
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def _ensure_feature_alignment(
    reference_columns: Sequence[str],
    bundle_features: np.ndarray,
    bundle_columns: Sequence[str],
) -> np.ndarray:
    """Ensure validation/test features align with the training column order."""
    if list(reference_columns) == list(bundle_columns):
        return bundle_features

    column_to_index = {column: idx for idx, column in enumerate(bundle_columns)}
    aligned = np.zeros(
        (bundle_features.shape[0], len(reference_columns)),
        dtype=bundle_features.dtype,
    )
    for out_idx, column in enumerate(reference_columns):
        src_idx = column_to_index.get(column)
        if src_idx is not None:
            aligned[:, out_idx] = bundle_features[:, src_idx]
    return aligned


def _evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    target_names: Sequence[str],
) -> Dict[str, float]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for features, target, _ in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model(features)
            preds.append(prediction.cpu())
            targets.append(target.cpu())

    if not preds:
        return {}
    pred_tensor = torch.cat(preds, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    return compute_metrics(pred_tensor, target_tensor, target_names)


def _prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_cfg: DatasetConfig,
) -> Tuple[AnalyticsDataset, AnalyticsDataset, Dict[str, object]]:
    train_bundle = extract_numeric_features(train_df, dataset_cfg)
    val_bundle = extract_numeric_features(val_df, dataset_cfg)

    val_bundle.features = _ensure_feature_alignment(
        train_bundle.feature_columns,
        val_bundle.features,
        val_bundle.feature_columns,
    )
    val_bundle.feature_columns = train_bundle.feature_columns

    feature_mean, feature_std = compute_feature_stats(train_bundle.features)
    train_dataset = AnalyticsDataset(
        train_bundle,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    val_dataset = AnalyticsDataset(
        val_bundle,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    feature_metadata = {
        "feature_columns": train_bundle.feature_columns,
        "target_columns": train_bundle.target_columns,
        "id_columns": train_bundle.id_columns,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }

    return train_dataset, val_dataset, feature_metadata


def _default_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}"


def _create_graph_data(dataset: AnalyticsDataset) -> List[Data]:
    """Convert a standard dataset into a list of graph Data objects."""
    graphs = []
    for i in range(len(dataset)):
        features, target, metadata = dataset[i]
        # In a real scenario, you would create a meaningful edge_index based on player proximity.
        # For this example, we'll use a fully-connected graph for simplicity.
        num_nodes = features.shape[0]
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        graphs.append(Data(x=features, edge_index=edge_index, y=target, **metadata))
    return graphs

def _train_on_split(
    config: TrainingConfig,
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_dir: Path,
) -> Dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, feature_metadata = _prepare_datasets(
        train_df,
        val_df,
        config.dataset,
    )

    is_graph_model = config.model.architecture == 'gat'

    if is_graph_model:
        train_graphs = _create_graph_data(train_dataset)
        val_graphs = _create_graph_data(val_dataset)
        train_loader = GraphDataLoader(train_graphs, batch_size=config.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs, batch_size=config.batch_size, shuffle=False)
        input_dim = train_graphs[0].x.shape[1]
    else:
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
        )
        input_dim = train_dataset.features.shape[1]

    output_dim = len(config.dataset.target_columns)
    model = build_baseline_model(
        input_dim=input_dim,
        output_dim=output_dim,
        model_cfg=config.model,
    )

    accelerator, device = _determine_device(config.device)
    module = AnalyticsLightningModule(
        model=model,
        optimizer_cfg=config.optimizer,
        target_names=config.dataset.target_columns,
        is_graph_module=is_graph_model,
    )

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:02d}-rmse{val_rmse:.4f}",
        monitor="val_rmse",
        mode="min",
        save_top_k=max(int(config.snapshot_count), 1),
        save_last=True,
    )
    callbacks = [checkpoint_callback]
    if config.optimizer.use_scheduler:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if config.early_stopping_patience is not None and config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_rmse",
                mode="min",
                patience=config.early_stopping_patience,
                verbose=True,
            )
        )

    use_logger = bool(config.optimizer.use_scheduler)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        default_root_dir=str(run_dir),
        accelerator=accelerator,
        devices=1,
        logger=use_logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        enable_model_summary=False,
        gradient_clip_val=config.gradient_clip_norm,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_checkpoint = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    snapshot_paths: List[str] = []
    if config.snapshot_count > 0 and checkpoint_callback.best_k_models:
        ordered = list(checkpoint_callback.best_k_models.keys())
        snapshot_paths = ordered[: config.snapshot_count]

    if snapshot_paths:
        LOGGER.info("Averaging %d checkpoints.", len(snapshot_paths))
        averaged_state = average_checkpoints(snapshot_paths)
        module.load_state_dict(averaged_state["state_dict"])
    elif best_checkpoint:
        LOGGER.info("Loading best checkpoint: %s", best_checkpoint)
        state = torch.load(best_checkpoint, map_location="cpu")
        module.load_state_dict(state["state_dict"])

    module.model.to(device)
    module.model.eval()

    val_summary = module.history[-1] if module.history else {}
    val_metrics = {
        "rmse": float(val_summary.get("val_rmse", float("nan"))),
        "mae": float(val_summary.get("val_mae", float("nan"))),
    }
    for name in config.dataset.target_columns:
        mae_key = f"val_mae_{name}"
        rmse_key = f"val_rmse_{name}"
        if mae_key in val_summary:
            val_metrics[f"mae_{name}"] = float(val_summary[mae_key])
        if rmse_key in val_summary:
            val_metrics[f"rmse_{name}"] = float(val_summary[rmse_key])

    torch.save(module.model.state_dict(), run_dir / "best_model.pt")

    with (run_dir / "config.json").open("w") as fp:
        json.dump(config.to_dict(), fp, indent=2)

    with (run_dir / "feature_metadata.json").open("w") as fp:
        json.dump(feature_metadata, fp, indent=2)

    with (run_dir / "history.json").open("w") as fp:
        json.dump(module.history, fp, indent=2)

    if val_metrics:
        with (run_dir / "validation_metrics.json").open("w") as fp:
            json.dump(val_metrics, fp, indent=2)

    return {
        "run_dir": str(run_dir),
        "val_metrics": val_metrics,
        "best_val_rmse": float(module.best_val_rmse),
        "history": module.history,
    }


def run_experiment(
    config: TrainingConfig,
    *,
    weeks: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    set_seed(config.seed)
    config.experiment_root.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading training dataframe ...")
    dataframe = load_training_dataframe(
        config.dataset,
        weeks=weeks,
        feature_cfg=config.features,
    )
    LOGGER.info("Adding player kinematics features...")
    dataframe = add_player_kinematics(dataframe)
    dataframe = add_distance_to_ball(dataframe)
    LOGGER.info("Adding relational features...")
    dataframe = add_relational_features(dataframe)

    train_df, val_df = split_train_validation(dataframe, config)
    run_name = config.run_name or _default_run_name("run")
    run_dir = config.experiment_root / run_name

    LOGGER.info("Starting training: %s", run_dir)
    result = _train_on_split(
        config,
        train_df=train_df,
        val_df=val_df,
        run_dir=run_dir,
    )
    return result


def run_cross_validation(
    config: TrainingConfig,
    *,
    n_folds: int,
    weeks: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, object]:
    if n_folds < 2:
        raise ValueError("Cross-validation requires n_folds >= 2.")

    set_seed(config.seed)
    config.experiment_root.mkdir(parents=True, exist_ok=True)

    dataframe = load_training_dataframe(
        config.dataset,
        weeks=weeks,
        feature_cfg=config.features,
    )
    LOGGER.info("Adding player kinematics features...")
    dataframe = add_player_kinematics(dataframe)
    dataframe = add_distance_to_ball(dataframe)
    LOGGER.info("Adding relational features...")
    dataframe = add_relational_features(dataframe)
    fold_column = config.dataset.fold_column

    if fold_column and fold_column in dataframe.columns:
        splitter = GroupKFold(n_splits=n_folds)
        groups = dataframe[fold_column].to_numpy()
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=config.seed)
        groups = None

    results: List[Dict[str, object]] = []
    fold_metrics: List[float] = []
    fold_dirs: List[str] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(dataframe, groups=groups), start=1):
        LOGGER.info("Fold %d/%d", fold_index, n_folds)
        train_df = dataframe.iloc[train_idx].reset_index(drop=True)
        val_df = dataframe.iloc[val_idx].reset_index(drop=True)

        fold_run_name = config.run_name or "cv"
        fold_run_dir = config.experiment_root / f"{fold_run_name}-fold{fold_index}"
        fold_config = replace(config, run_name=f"{fold_run_name}-fold{fold_index}")

        fold_result = _train_on_split(
            fold_config,
            train_df=train_df,
            val_df=val_df,
            run_dir=fold_run_dir,
        )
        results.append(fold_result)
        fold_dirs.append(fold_result["run_dir"])
        fold_metrics.append(float(fold_result.get("best_val_rmse", float("nan"))))

    mean_rmse = float(np.nanmean(fold_metrics)) if fold_metrics else float("nan")
    std_rmse = float(np.nanstd(fold_metrics)) if fold_metrics else float("nan")

    payload = {
        "fold_results": results,
        "fold_rmse": fold_metrics,
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
        "fold_dirs": fold_dirs,
    }
    return payload


def average_checkpoints(
    paths: Sequence[Union[str, Path]]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Average Lightning checkpoint state dictionaries."""
    if not paths:
        raise ValueError("No checkpoint paths provided.")

    avg_state: Dict[str, torch.Tensor] = {}
    count = 0

    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("state_dict")
        if state is None:
            raise KeyError(f"Checkpoint at {path} missing 'state_dict'.")
        if not avg_state:
            avg_state = {key: tensor.clone() for key, tensor in state.items()}
        else:
            for key, tensor in state.items():
                avg_state[key] += tensor
        count += 1

    for key in avg_state:
        avg_state[key] /= count
    return {"state_dict": avg_state}


def _parse_int_list(value: Optional[str], *, default: Iterable[int]) -> List[int]:
    if not value:
        return list(default)
    parts = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(int(chunk))
    return parts


def _parse_weeks(value: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    if not value:
        return None
    weeks: List[Tuple[int, int]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "_" in token:
            season_part, week_part = token.split("_", 1)
        elif "-w" in token.lower():
            season_part, week_part = token.lower().split("-w", 1)
        else:
            raise ValueError(f"Week token '{token}' is not in season_wXX format.")
        season = int(season_part)
        week = int(week_part.replace("w", ""))
        weeks.append((season, week))
    return weeks


def build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    dataset_cfg = DatasetConfig(
        data_root=Path(args.data_root),
        bundle_dirname=args.bundle_dirname,
        target_columns=["target_x", "target_y"],
        include_supplementary=not args.no_supplementary,
        fold_column=args.fold_column,
    )

    model_cfg = ModelConfig(
        architecture=args.architecture,
        hidden_dims=_parse_int_list(args.hidden_dims, default=[512, 256]),
        dropout=args.dropout,
        n_heads=args.n_heads,
        sequence_length=args.sequence_length,
        use_metadata=False,
        num_layers=args.model_layers,
        latent_dim=args.latent_dim,
        num_latents=args.num_latents,
    )

    feature_cfg = FeatureConfig(
        use_pairwise_distance=args.use_pairwise_distance,
        use_game_clock_seconds=args.use_game_clock_seconds,
    )

    optimizer_cfg = OptimizerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_scheduler=not args.no_scheduler,
        cosine_t_max=args.cosine_t_max,
        warmup_steps=args.warmup_steps,
    )

    training_cfg = TrainingConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        features=feature_cfg,
        optimizer=optimizer_cfg,
        experiment_root=Path(args.output_dir),
        run_name=args.run_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        epochs=args.epochs,
        gradient_clip_norm=args.gradient_clip,
        seed=args.seed,
        device=args.device,
        val_fraction=args.val_fraction,
        snapshot_count=args.snapshot_count,
        early_stopping_patience=args.early_stopping_patience,
    )
    return training_cfg


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NFL analytics baseline.")
    parser.add_argument("--data-root", default="competitions/nfl_big_data_bowl_2026_analytics/data/raw")
    parser.add_argument("--bundle-dirname", default="nfl-big-data-bowl-2026-analytics")
    parser.add_argument("--output-dir", default="competitions/nfl_big_data_bowl_2026_analytics/outputs/baseline")
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-count", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--architecture", type=str, default="mlp", choices=["mlp", "residual_mlp", "perceiver", "gat"])
    parser.add_argument("--hidden-dims", type=str, default="512,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--model-layers", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-latents", type=int, default=8)
    parser.add_argument("--cosine-t-max", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--no-supplementary", action="store_true")
    parser.add_argument("--use-pairwise-distance", action="store_true")
    parser.add_argument("--use-game-clock-seconds", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--fold-column", default="game_id")
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--weeks", type=str, help="Comma-separated list like 2023_01,2023_02.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    config = build_config_from_args(args)
    weeks = _parse_weeks(args.weeks)

    if args.n_folds > 1:
        LOGGER.info("Running %d-fold cross-validation.", args.n_folds)
        result = run_cross_validation(
            config,
            n_folds=args.n_folds,
            weeks=weeks,
        )
    else:
        result = run_experiment(config, weeks=weeks)

    LOGGER.info("Training complete:\n%s", json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
