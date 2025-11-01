"""Time-series aware cross-validation for NFL trajectory prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold

from ..data.loaders import available_train_weeks


def create_temporal_folds(
    weeks: List[Tuple[int, int]],
    n_folds: int = 5,
    strategy: str = "sequential",
) -> List[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """Create time-series aware folds.
    
    Args:
        weeks: List of (season, week) tuples
        n_folds: Number of folds
        strategy: 'sequential' or 'interleaved'
    
    Returns:
        List of (train_weeks, val_weeks) tuples
    """
    weeks = sorted(weeks)
    
    if strategy == "sequential":
        # Sequential: train on earlier weeks, validate on later
        fold_size = len(weeks) // n_folds
        folds = []
        
        for i in range(n_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < n_folds - 1 else len(weeks)
            
            val_weeks = weeks[val_start:val_end]
            train_weeks = weeks[:val_start] + weeks[val_end:]
            
            folds.append((train_weeks, val_weeks))
    
    elif strategy == "interleaved":
        # Interleaved: distribute weeks across folds
        folds = [[] for _ in range(n_folds)]
        
        for i, week in enumerate(weeks):
            folds[i % n_folds].append(week)
        
        result = []
        for i in range(n_folds):
            val_weeks = folds[i]
            train_weeks = [w for j, fold in enumerate(folds) if j != i for w in fold]
            result.append((train_weeks, val_weeks))
        
        folds = result
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return folds


def run_cross_validation(
    model_factory: Callable,
    train_fn: Callable,
    evaluate_fn: Callable,
    *,
    n_folds: int = 5,
    strategy: str = "sequential",
    output_dir: Path,
    base_dir: Path = None,
) -> Dict[str, List[float]]:
    """Run time-series aware cross-validation.
    
    Args:
        model_factory: Function that returns a new model instance
        train_fn: Function(model, train_weeks) -> trained_model
        evaluate_fn: Function(model, val_weeks) -> metrics_dict
        n_folds: Number of folds
        strategy: Folding strategy
        output_dir: Output directory
        base_dir: Data directory
    
    Returns:
        Dictionary of metrics per fold
    """
    # Get available weeks
    weeks = available_train_weeks(base_dir=base_dir)
    print(f"Found {len(weeks)} training weeks")
    
    # Create folds
    folds = create_temporal_folds(weeks, n_folds=n_folds, strategy=strategy)
    
    fold_metrics = {
        "mae": [],
        "fde": [],
        "ade": [],
    }
    
    for fold_idx, (train_weeks, val_weeks) in enumerate(folds, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{n_folds}")
        print(f"Train weeks: {len(train_weeks)}, Val weeks: {len(val_weeks)}")
        print(f"{'='*60}")
        
        # Create model
        model = model_factory()
        
        # Train
        print("Training...")
        model = train_fn(model, train_weeks)
        
        # Evaluate
        print("Evaluating...")
        metrics = evaluate_fn(model, val_weeks)
        
        # Store metrics
        for key in fold_metrics:
            if key in metrics:
                fold_metrics[key].append(metrics[key])
        
        # Save fold model
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(model, 'state_dict'):
            torch.save(model.state_dict(), fold_dir / "model.pt")
        
        print(f"Fold {fold_idx} metrics: {metrics}")
    
    # Calculate summary
    summary = {}
    for metric_name, values in fold_metrics.items():
        if values:
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
    
    # Save results
    results_path = output_dir / "cv_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "fold_metrics": fold_metrics,
            "summary": summary,
            "n_folds": n_folds,
            "strategy": strategy,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    
    return fold_metrics


__all__ = [
    "create_temporal_folds",
    "run_cross_validation",
]

