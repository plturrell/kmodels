"""Dynamic stacking ensemble with adaptive weights and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from .adaptive_ensemble import AdaptiveEnsemble
from .meta_features import MetaFeatureBuilder
from .regime_detection import RegimeDetector
from ..training.walk_forward_cv import TimeSeriesCV


class DynamicStacker:
    """Dynamic stacking ensemble with adaptive weighting."""
    
    def __init__(
        self,
        base_models: Optional[List] = None,
        meta_model_type: str = "linear",
        adaptive_ensemble: Optional[AdaptiveEnsemble] = None,
        regime_detector: Optional[RegimeDetector] = None,
        meta_feature_builder: Optional[MetaFeatureBuilder] = None,
        blend_factor: float = 0.7,
    ):
        """Initialize dynamic stacker.
        
        Args:
            base_models: List of base models (optional, can be loaded from checkpoints)
            meta_model_type: Type of meta-model ('linear' or 'gbm')
            adaptive_ensemble: Adaptive ensemble instance
            regime_detector: Regime detector instance
            meta_feature_builder: Meta-feature builder instance
            blend_factor: Factor for blending meta-model with adaptive weights (0-1)
        """
        self.base_models = base_models or []
        self.meta_model_type = meta_model_type
        self.adaptive_ensemble = adaptive_ensemble
        self.regime_detector = regime_detector
        self.meta_feature_builder = meta_feature_builder
        self.blend_factor = blend_factor
        
        # Initialize meta-model
        if meta_model_type == "linear":
            self.meta_model = LinearRegression()
        elif meta_model_type == "gbm":
            self.meta_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown meta_model_type: {meta_model_type}")
        
        # Store feature columns for alignment
        self.feature_columns: Optional[List[str]] = None
        
    def load_models_from_checkpoints(
        self,
        checkpoint_dirs: List[Path],
        model_factory: Callable,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> List[nn.Module]:
        """Load PyTorch models from checkpoint directories.
        
        Args:
            checkpoint_dirs: List of checkpoint directory paths
            model_factory: Function that creates a model instance
            device: Device to load models on
            
        Returns:
            List of loaded models
        """
        models = []
        
        for ckpt_dir in checkpoint_dirs:
            ckpt_dir = Path(ckpt_dir)
            
            # Look for checkpoint files
            checkpoint_candidates = [
                ckpt_dir / "model_state.pt",
                ckpt_dir / "checkpoints" / "last.ckpt",
                ckpt_dir / "best_model.pt",
            ]
            
            checkpoint_path = None
            for candidate in checkpoint_candidates:
                if candidate.exists():
                    checkpoint_path = candidate
                    break
            
            if checkpoint_path is None:
                # Try glob pattern
                glob_matches = list(ckpt_dir.glob("**/*.pt")) + list(ckpt_dir.glob("**/*.ckpt"))
                if glob_matches:
                    checkpoint_path = glob_matches[0]
                else:
                    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
            
            # Load model
            model = model_factory()
            state = torch.load(checkpoint_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(state, dict):
                state_dict = state.get("state_dict", state)
            else:
                state_dict = state
            
            # Remove "model." prefix if present (Lightning format)
            if isinstance(state_dict, dict):
                state_dict = {
                    k.replace("model.", ""): v 
                    for k, v in state_dict.items()
                }
            
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            model.eval()
            models.append(model)
        
        return models
    
    def generate_oof_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: List[nn.Module],
        cv: TimeSeriesCV,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 512,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate out-of-fold predictions using walk-forward CV.
        
        Args:
            X: Feature dataframe
            y: Target series
            models: List of models to evaluate
            cv: TimeSeriesCV instance for splits
            device: Device for inference
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (OOF predictions DataFrame, actuals Series)
        """
        n_samples = len(X)
        n_models = len(models)
        
        # Initialize OOF prediction arrays
        oof_predictions = np.zeros((n_samples, n_models))
        oof_actuals = np.zeros(n_samples)
        oof_indices = np.zeros(n_samples, dtype=bool)
        
        # Generate predictions for each CV fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Convert to tensors
            X_val_tensor = torch.FloatTensor(X_val_fold.values).to(device)
            
            # Get predictions from each model
            for model_idx, model in enumerate(models):
                with torch.no_grad():
                    preds = model(X_val_tensor)
                    
                    # Handle variance output (heteroscedastic)
                    if isinstance(preds, tuple):
                        preds = preds[0]  # Use mean prediction
                    
                    preds = preds.cpu().numpy().flatten()
                    
                    # Store predictions
                    oof_predictions[val_idx, model_idx] = preds
            
            # Store actuals
            oof_actuals[val_idx] = y_val_fold.values
            oof_indices[val_idx] = True
        
        # Create DataFrame with OOF predictions
        oof_df = pd.DataFrame(
            oof_predictions,
            index=X.index,
            columns=[f"model_{i}" for i in range(n_models)]
        )
        
        oof_actuals_series = pd.Series(oof_actuals, index=X.index)
        
        return oof_df, oof_actuals_series
    
    def fit(
        self,
        base_predictions: pd.DataFrame,
        original_features: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        id_column: Optional[pd.Series] = None,
    ) -> None:
        """Fit the stacking ensemble.
        
        Args:
            base_predictions: DataFrame with base model OOF predictions
            original_features: Original feature dataframe (for meta-features)
            y: Target values (required for training)
            id_column: ID column for alignment (optional)
        """
        if y is None:
            raise ValueError("Target values (y) required for fitting")
        
        # Align indices
        common_idx = base_predictions.index.intersection(y.index)
        if len(common_idx) < len(base_predictions):
            base_predictions = base_predictions.loc[common_idx]
            y = y.loc[common_idx]
            if original_features is not None:
                original_features = original_features.loc[common_idx]
        
        # Register models in adaptive ensemble
        if self.adaptive_ensemble is not None:
            # Detect current regime if regime detector available
            current_regime = None
            if self.regime_detector is not None and original_features is not None:
                regimes = self.regime_detector.detect_regime(original_features)
                current_regime = regimes.iloc[-1] if len(regimes) > 0 else None
            
            # Register each model
            for col in base_predictions.columns:
                model_id = col
                pred_series = base_predictions[col]
                
                # Align predictions and actuals
                aligned_idx = pred_series.index.intersection(y.index)
                pred_aligned = pred_series.loc[aligned_idx]
                actual_aligned = y.loc[aligned_idx]
                
                self.adaptive_ensemble.register_model(
                    model_id=model_id,
                    predictions=pred_aligned,
                    actuals=actual_aligned,
                    current_regime=current_regime,
                )
        
        # Build enhanced meta-features
        if self.meta_feature_builder is not None:
            X_meta = self.meta_feature_builder.build_meta_features(
                base_predictions=base_predictions,
                original_features=original_features,
                id_column=id_column,
            )
        else:
            # Use base predictions only
            X_meta = base_predictions.copy()
        
        # Store feature columns for prediction alignment
        self.feature_columns = list(X_meta.columns)
        
        # Train meta-model
        X_meta_aligned = X_meta.loc[y.index]
        y_aligned = y.loc[X_meta_aligned.index]
        
        self.meta_model.fit(X_meta_aligned.values, y_aligned.values)
    
    def predict(
        self,
        base_predictions: pd.DataFrame,
        original_features: Optional[pd.DataFrame] = None,
        id_column: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Generate predictions using stacking ensemble.
        
        Args:
            base_predictions: DataFrame with base model predictions
            original_features: Original feature dataframe (for meta-features)
            id_column: ID column for alignment (optional)
            
        Returns:
            Series with final predictions
        """
        # Build meta-features
        if self.meta_feature_builder is not None:
            X_meta = self.meta_feature_builder.build_meta_features(
                base_predictions=base_predictions,
                original_features=original_features,
                id_column=id_column,
            )
        else:
            X_meta = base_predictions.copy()
        
        # Ensure feature alignment
        if self.feature_columns is not None:
            # Add missing columns with zeros
            for col in self.feature_columns:
                if col not in X_meta.columns:
                    X_meta[col] = 0.0
            # Reorder to match training
            X_meta = X_meta[self.feature_columns]
        
        # Get meta-model predictions
        meta_preds = self.meta_model.predict(X_meta.values)
        meta_preds_series = pd.Series(meta_preds, index=X_meta.index)
        
        # Get adaptive weights if available
        if self.adaptive_ensemble is not None:
            current_regime = None
            if self.regime_detector is not None and original_features is not None:
                regimes = self.regime_detector.detect_regime(original_features)
                current_regime = regimes.iloc[-1] if len(regimes) > 0 else None
            
            adaptive_weights = self.adaptive_ensemble.calculate_adaptive_weights(
                current_regime=current_regime
            )
            
            # Blend meta-model with adaptive-weighted ensemble
            adaptive_pred = pd.Series(0.0, index=base_predictions.index)
            for col in base_predictions.columns:
                if col in adaptive_weights:
                    adaptive_pred += adaptive_weights[col] * base_predictions[col]
            
            # Blend: blend_factor * meta_model + (1 - blend_factor) * adaptive
            final_preds = (
                self.blend_factor * meta_preds_series +
                (1 - self.blend_factor) * adaptive_pred
            )
        else:
            final_preds = meta_preds_series
        
        return final_preds

