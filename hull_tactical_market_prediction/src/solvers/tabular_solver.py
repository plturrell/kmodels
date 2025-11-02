"""Neural solver orchestrating training and inference."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..config.experiment import ExperimentConfig
from ..modeling.tabular import TabularRegressor
from ..modeling.perceiver import PerceiverRegressor
from ..modeling.losses import LossFactory
import logging

from ..training.datamodule import PreparedFeatures, create_dataloaders, prepare_features
from ..training.curriculum import rank_samples_by_difficulty, create_curriculum_sampler_weights
from torch.utils.data import WeightedRandomSampler

LOGGER = logging.getLogger(__name__)


def _prepare_output_dir(base_dir: Path, run_name: Optional[str] = None) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        timestamp = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
        run_dir = base_dir / timestamp
    else:
        run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _serialize_feature_stats(run_dir: Path, prepared: PreparedFeatures) -> None:
    stats_path = run_dir / "feature_stats.json"
    payload = {
        "columns": list(prepared.train_features.columns),
        "means": prepared.feature_means.astype(float).tolist(),
        "stds": prepared.feature_stds.astype(float).tolist(),
    }
    stats_path.write_text(json.dumps(payload, indent=2))


def _serialize_config(run_dir: Path, config: ExperimentConfig) -> None:
    (run_dir / "experiment_config.json").write_text(json.dumps(config.to_dict(), indent=2))


def _serialize_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def run_experiment(
    config: ExperimentConfig,
    *,
    run_name: Optional[str] = None,
    checkpoint: Optional[Path] = None,
) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_features(config)
    train_loader, val_loader, test_loader, val_indices = create_dataloaders(
        prepared, config.trainer, curriculum_cfg=config.curriculum, epoch=0
    )

    # Enable variance output for heteroscedastic loss
    output_variance = config.loss.mode == "heteroscedastic"
    
    if config.model.model_type == "perceiver":
        model = PerceiverRegressor(
            input_dim=prepared.train_features.shape[1],
            num_latents=config.model.perceiver_num_latents,
            latent_dim=config.model.perceiver_latent_dim,
            num_layers=config.model.perceiver_layers,
            num_heads=config.model.perceiver_heads,
            dropout=config.model.perceiver_dropout,
            ff_mult=config.model.perceiver_ff_mult,
        ).to(device)
    else:
        model = TabularRegressor(
            input_dim=prepared.train_features.shape[1],
            hidden_dims=config.model.hidden_dims,
            dropout=config.model.dropout,
            activation=config.model.activation,
            batch_norm=config.model.batch_norm,
            output_variance=output_variance,
        ).to(device)

    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict)

    loss_fn = LossFactory(config.loss)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps,
    )

    run_dir = _prepare_output_dir(config.output_dir, run_name)
    _serialize_config(run_dir, config)
    _serialize_feature_stats(run_dir, prepared)

    history, best_state, best_rmse = _train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        config.trainer.epochs,
        config.optimizer.gradient_clip_val,
        loss_fn,
        prepared=prepared,
        config=config,
    )

    metrics = {
        "best_val_rmse": best_rmse,
        "history": history,
    }
    _serialize_metrics(run_dir, metrics)

    model.load_state_dict(best_state)

    model_path = run_dir / "model_state.pt"
    torch.save({"state_dict": model.state_dict(), "config": config.to_dict()}, model_path)

    # Save validation predictions for analysis
    if val_indices.size > 0:
        val_features_df = prepared.train_features.iloc[val_indices]
        val_ids = prepared.train_ids.iloc[val_indices]
        val_targets = prepared.train_target.iloc[val_indices]
        model.eval()
        with torch.no_grad():
            val_tensor = torch.from_numpy(val_features_df.to_numpy(dtype=np.float32)).to(device)
            val_preds = model(val_tensor).cpu().numpy()
        oof_df = pd.DataFrame(
            {
                config.id_column: val_ids.to_numpy(),
                f"{config.target_column}_oof": val_preds.flatten(),
            }
        )
        oof_path = run_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        LOGGER.info("Saved out-of-fold predictions to %s", oof_path)

    if config.test_csv and test_loader is not None:
        preds = _predict(model, test_loader, device)
        submission = _build_submission(config, prepared, preds)
        submission_path = run_dir / "submission.csv"
        submission.to_csv(submission_path, index=False)
        latest_path = config.output_dir / "latest_submission.csv"
        submission.to_csv(latest_path, index=False)

    return run_dir


def _train_model(
    model: TabularRegressor,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    grad_clip: float,
    loss_fn: LossFactory,
    prepared: Optional[PreparedFeatures] = None,
    config: Optional[ExperimentConfig] = None,
) -> Tuple[List[dict], dict, float]:
    best_rmse = float("inf")
    best_state: dict = {}
    history: List[dict] = []
    
    # Pre-compute difficulty ranks if curriculum is enabled (for efficiency)
    curriculum_difficulty_ranks = None
    train_indices_stored = None
    
    if config and config.curriculum.enabled and prepared is not None:
        # Store train/val split indices for consistent curriculum application
        num_samples = len(prepared.train_features)
        val_size = max(1, int(num_samples * config.trainer.val_fraction))
        train_size = num_samples - val_size
        
        if config.trainer.time_series_holdout:
            train_indices_stored = np.arange(0, train_size)
        else:
            # For shuffled, recreate the same permutation
            torch.manual_seed(config.seed)
            permutation = torch.randperm(num_samples)
            train_indices_stored = permutation[:train_size].detach().cpu().numpy()
        
        # Compute difficulty only on training subset
        train_features_df = prepared.train_features.iloc[train_indices_stored]
        curriculum_difficulty_ranks = rank_samples_by_difficulty(
            train_features_df,
            difficulty_metric=config.curriculum.difficulty_metric,
        )
        LOGGER.info("Pre-computed curriculum difficulty ranks for %d training samples", len(curriculum_difficulty_ranks))

    for epoch in range(1, epochs + 1):
        # Recreate train_loader with updated curriculum weights if curriculum is enabled
        if config and config.curriculum.enabled and prepared is not None and curriculum_difficulty_ranks is not None and train_indices_stored is not None:
            weights = create_curriculum_sampler_weights(curriculum_difficulty_ranks, epoch, config.curriculum)
            
            # Recreate train dataset using stored indices
            features_tensor = torch.from_numpy(prepared.train_features.to_numpy(dtype=np.float32))
            target_tensor = torch.from_numpy(prepared.train_target.to_numpy(dtype=np.float32)).unsqueeze(-1)
            
            train_features = features_tensor[train_indices_stored]
            train_targets = target_tensor[train_indices_stored]
            
            # Normalize weights
            weights = weights / (weights.sum() + 1e-8)
            weights_tensor = torch.from_numpy(weights.astype(np.float32))
            
            from torch.utils.data import TensorDataset
            train_dataset = TensorDataset(train_features, train_targets)
            sampler = WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(weights),
                replacement=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.trainer.batch_size,
                sampler=sampler,
                num_workers=config.trainer.num_workers,
                pin_memory=True,
            )
            if epoch == 1 or epoch % 5 == 0:  # Log periodically
                # Calculate current curriculum coverage
                if epoch < config.curriculum.min_difficulty_epoch:
                    coverage_pct = config.curriculum.start_ratio * 100
                elif epoch >= config.curriculum.max_difficulty_epoch:
                    coverage_pct = 100.0
                else:
                    progress = (epoch - config.curriculum.min_difficulty_epoch) / max(1, config.curriculum.max_difficulty_epoch - config.curriculum.min_difficulty_epoch)
                    coverage_pct = (config.curriculum.start_ratio + (1.0 - config.curriculum.start_ratio) * progress) * 100
                LOGGER.info("Epoch %d: Curriculum sampling %.1f%% of training samples (easiest → hardest)", 
                           epoch, coverage_pct)
        
        train_loss, train_rmse = _run_epoch(
            model, optimizer, train_loader, device, grad_clip, loss_fn, config, prepared
        )
        val_loss, val_rmse = _evaluate(model, val_loader, device, loss_fn, config, prepared)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
            }
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return history, best_state, best_rmse


def _run_epoch(
    model: TabularRegressor,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    grad_clip: float,
    loss_fn: LossFactory,
    config: Optional[ExperimentConfig] = None,
    prepared: Optional[PreparedFeatures] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_squared_error = 0.0
    total_samples = 0
    
    output_variance = config is not None and config.loss.mode == "heteroscedastic"
    use_adaptive_dropout = config is not None and config.curriculum.enabled and config.curriculum.adaptive_dropout
    
    # Get uncertainties if available for heteroscedastic loss
    uncertainties_tensor = None
    if prepared is not None and prepared.target_uncertainties is not None:
        uncertainties_tensor = torch.from_numpy(prepared.target_uncertainties).to(device)

    for batch_idx, (features, target) in enumerate(dataloader):
        features = features.to(device)
        target = target.to(device).squeeze(-1)
        
        # Adaptive dropout: scale by sample uncertainty if enabled
        if use_adaptive_dropout and uncertainties_tensor is not None:
            # Get batch uncertainties (approximate - would need proper indexing)
            # For now, apply uniform scaling based on mean uncertainty
            mean_uncertainty = torch.mean(uncertainties_tensor).item()
            # Scale dropout: higher uncertainty -> higher dropout
            dropout_scale = min(2.0, 1.0 + mean_uncertainty)
            # Note: This is a simplified version; full implementation would need per-sample dropout
            # which requires modifying model architecture

        optimizer.zero_grad()
        
        if output_variance:
            preds, pred_var = model(features)
            # Get batch uncertainties (simplified - would need proper batch indexing)
            batch_uncertainties = None
            if uncertainties_tensor is not None:
                # For now, use mean uncertainty (full implementation needs proper indexing)
                batch_size = features.size(0)
                batch_uncertainties = torch.full((batch_size,), torch.mean(uncertainties_tensor), device=device)
            loss = loss_fn(preds, target, pred_variance=pred_var, target_uncertainty=batch_uncertainties)
        else:
            preds = model(features)
            loss = loss_fn(preds, target)

        loss.backward()
        if grad_clip > 0:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        
        # For RMSE, use mean prediction (first element if tuple)
        preds_mean = preds[0] if isinstance(preds, tuple) else preds
        total_squared_error += F.mse_loss(preds_mean, target, reduction="sum").item()
        total_samples += batch_size

    mean_loss = total_loss / total_samples
    rmse = float(np.sqrt(total_squared_error / total_samples))
    return mean_loss, rmse


def _evaluate(
    model: TabularRegressor,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: LossFactory,
    config: Optional[ExperimentConfig] = None,
    prepared: Optional[PreparedFeatures] = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_squared_error = 0.0
    total_samples = 0
    
    output_variance = config is not None and config.loss.mode == "heteroscedastic"
    uncertainties_tensor = None
    if prepared is not None and prepared.target_uncertainties is not None:
        uncertainties_tensor = torch.from_numpy(prepared.target_uncertainties).to(device)

    with torch.no_grad():
        for features, target in dataloader:
            features = features.to(device)
            target = target.to(device).squeeze(-1)
            
            if output_variance:
                preds, pred_var = model(features)
                batch_uncertainties = None
                if uncertainties_tensor is not None:
                    batch_size = features.size(0)
                    batch_uncertainties = torch.full((batch_size,), torch.mean(uncertainties_tensor), device=device)
                loss = loss_fn(preds, target, pred_variance=pred_var, target_uncertainty=batch_uncertainties)
            else:
                preds = model(features)
                loss = loss_fn(preds, target)
            
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            
            preds_mean = preds[0] if isinstance(preds, tuple) else preds
            total_squared_error += F.mse_loss(preds_mean, target, reduction="sum").item()
            total_samples += batch_size

    mean_loss = total_loss / total_samples
    rmse = float(np.sqrt(total_squared_error / total_samples))
    return mean_loss, rmse


def _predict(model: TabularRegressor, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for (features,) in dataloader:
            features = features.to(device)
            outputs = model(features)
            # If model outputs (mean, variance), extract mean
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds.append(outputs.detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def _build_submission(config: ExperimentConfig, prepared: PreparedFeatures, preds: np.ndarray):
    import pandas as pd

    if prepared.test_ids is None:
        raise RuntimeError("Test IDs missing; cannot build submission.")

    if config.sample_submission:
        submission = pd.read_csv(config.sample_submission)
        target_cols = [col for col in submission.columns if col != config.id_column]
        if len(target_cols) != 1:
            raise ValueError("Sample submission must contain exactly one prediction column.")
        submission[target_cols[0]] = preds
        return submission

    return pd.DataFrame(
        {
            config.id_column: prepared.test_ids,
            config.submission_column: preds,
        }
    )
