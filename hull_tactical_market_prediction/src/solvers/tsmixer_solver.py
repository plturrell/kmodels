"""Solver for the TSMixer model."""

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
from ..modeling.tsmixer import TSMixer
from ..training.datamodule import PreparedFeatures, prepare_features
import logging

from ..training.tsmixer_datamodule import (
    TSMixerDataConfig,
    create_tsmixer_dataloaders,
)

LOGGER = logging.getLogger(__name__)


def _align_predictions(preds: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if preds.ndim == 3 and preds.size(-1) == 1:
        preds = preds.squeeze(-1)
    if target.ndim == 3 and target.size(-1) == 1:
        target = target.squeeze(-1)
    if preds.ndim == 2 and preds.size(-1) == 1:
        preds = preds.squeeze(-1)
    if target.ndim == 2 and target.size(-1) == 1:
        target = target.squeeze(-1)
    if preds.ndim == 1:
        preds = preds.unsqueeze(-1)
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    return preds, target


def _prepare_output_dir(base_dir: Path, run_name: Optional[str] = None) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_name is None:
        timestamp = datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
        run_dir = base_dir / timestamp
    else:
        run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_tsmixer_experiment(
    config: ExperimentConfig,
    data_config: TSMixerDataConfig,
    *,    run_name: Optional[str] = None,
    checkpoint: Optional[Path] = None,
) -> Path:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_features(config)
    train_loader, val_loader, test_loader = create_tsmixer_dataloaders(
        prepared, config.trainer, data_config
    )

    model = TSMixer(
        sequence_length=data_config.sequence_length,
        prediction_length=data_config.prediction_length,
        input_channels=prepared.train_features.shape[1],
        output_channels=1,  # We are predicting a single target
        num_blocks=config.model.perceiver_layers,  # Re-using perceiver config for now
        ff_dim=config.model.hidden_dims[0],
        dropout_rate=config.model.dropout,
    ).to(device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state.get("state_dict", state))

    loss_fn = F.mse_loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    run_dir = _prepare_output_dir(config.output_dir, run_name)

    history, best_state, best_rmse = _train_model(
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        config.trainer.epochs,
        loss_fn,
    )

    metrics = {"best_val_rmse": best_rmse, "history": history}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    model.load_state_dict(best_state)
    model_path = run_dir / "model_state.pt"
    torch.save({"state_dict": model.state_dict(), "config": config.to_dict()}, model_path)

    # Save validation predictions for analysis
    val_preds = _predict(model, val_loader, device)
    val_ids = prepared.train_ids.iloc[-len(val_preds):]
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
    model: TSMixer,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    loss_fn,
) -> Tuple[List[dict], dict, float]:
    best_rmse = float("inf")
    best_state: dict = {}
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_rmse = _run_epoch(model, optimizer, train_loader, device, loss_fn)
        val_loss, val_rmse = _evaluate(model, val_loader, device, loss_fn)
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
    model: TSMixer,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_squared_error = 0.0
    total_samples = 0

    for features, target in dataloader:
        features = features.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        preds = model(features)
        preds, target = _align_predictions(preds, target)
        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_squared_error += F.mse_loss(preds, target, reduction="sum").item()
        total_samples += target.numel()

    mean_loss = total_loss / total_samples
    rmse = float(np.sqrt(total_squared_error / total_samples))
    return mean_loss, rmse


def _evaluate(
    model: TSMixer, dataloader: DataLoader, device: torch.device, loss_fn
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_squared_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, target in dataloader:
            features = features.to(device)
            target = target.to(device)
            preds = model(features)
            preds, target = _align_predictions(preds, target)
            loss = loss_fn(preds, target)
            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_squared_error += F.mse_loss(preds, target, reduction="sum").item()
            total_samples += target.numel()

    mean_loss = total_loss / total_samples
    rmse = float(np.sqrt(total_squared_error / total_samples))
    return mean_loss, rmse


def _predict(model: TSMixer, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            outputs = model(features)
            outputs = outputs.detach().cpu()
            if outputs.ndim == 3 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            preds.append(outputs)
    return torch.cat(preds, dim=0).numpy().reshape(-1)


def _build_submission(config: ExperimentConfig, prepared: PreparedFeatures, preds: np.ndarray):
    if prepared.test_ids is None:
        raise RuntimeError("Test IDs missing; cannot build submission.")

    # The number of predictions will be smaller than the number of test IDs
    # because of the sequence-to-sequence nature of the model.
    num_preds = len(preds)
    test_ids = prepared.test_ids.iloc[-num_preds:]

    return pd.DataFrame({config.id_column: test_ids, config.submission_column: preds})
