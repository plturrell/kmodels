"""Adaptive ensemble weighting based on rolling Sharpe ratios and regime correlation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.financial_metrics import sharpe_ratio, max_drawdown, hit_rate


@dataclass
class ModelDiagnostics:
    """Track model performance and regime correlations."""
    
    model_id: str
    rolling_sharpe: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    recent_performance: float = 0.0
    regime_correlation: Dict[str, float] = field(default_factory=dict)
    last_active: Optional[datetime] = None
    prediction_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    actual_history: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    prediction_history_aligned: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    actual_history_aligned: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    max_dd: float = 0.0
    hit_rate_value: float = 0.0


class AdaptiveEnsemble:
    """Adaptive ensemble with regime-aware weighting."""
    
    def __init__(
        self,
        lookback_window: int = 63,
        min_active_periods: int = 21,
        sharpe_decay_factor: float = 0.95,
        min_weight_floor: float = 0.05,
    ):
        """Initialize adaptive ensemble.
        
        Args:
            lookback_window: Rolling window for Sharpe calculation (default 63 days)
            min_active_periods: Minimum periods required for model to be active (default 21)
            sharpe_decay_factor: Exponential decay factor for recency (default 0.95)
            min_weight_floor: Minimum weight per model (default 0.05)
        """
        self.lookback_window = lookback_window
        self.min_active_periods = min_active_periods
        self.sharpe_decay_factor = sharpe_decay_factor
        self.min_weight_floor = min_weight_floor
        
        self.models: Dict[str, ModelDiagnostics] = {}
    
    def register_model(
        self,
        model_id: str,
        predictions: pd.Series,
        actuals: pd.Series,
        current_regime: Optional[str] = None,
    ) -> None:
        """Register or update model with new predictions.
        
        Args:
            model_id: Unique identifier for the model
            predictions: Model predictions (aligned with actuals)
            actuals: Actual returns/targets (aligned with predictions)
            current_regime: Current market regime (optional)
        """
        if model_id not in self.models:
            self.models[model_id] = ModelDiagnostics(model_id=model_id)
        
        diag = self.models[model_id]
        
        # Update prediction and actual history
        if len(diag.prediction_history) == 0:
            diag.prediction_history = predictions.copy()
            diag.actual_history = actuals.copy()
        else:
            # Align by index and concatenate
            diag.prediction_history = pd.concat([diag.prediction_history, predictions])
            diag.actual_history = pd.concat([diag.actual_history, actuals])
            # Remove duplicates by index (keep last)
            diag.prediction_history = diag.prediction_history[~diag.prediction_history.index.duplicated(keep='last')]
            diag.actual_history = diag.actual_history[~diag.actual_history.index.duplicated(keep='last')]
        
        # Calculate strategy returns from predictions and actuals
        # For return prediction models: strategy return = prediction * actual_return
        # This assumes predictions are signed return predictions (can be used directly)
        # Alternative interpretation: if predictions are signals, use sign(pred) * actual
        # We use direct multiplication to preserve magnitude information
        aligned_idx = predictions.index.intersection(actuals.index)
        pred_aligned = predictions.loc[aligned_idx]
        actual_aligned = actuals.loc[aligned_idx]
        
        # Strategy returns: use predictions as position sizing
        # If prediction and actual have same sign, positive return; else negative
        # For return predictions: strategy_return = sign(prediction) * actual
        # This captures directional accuracy with magnitude
        returns = pd.Series(
            np.sign(pred_aligned.values) * actual_aligned.values,
            index=aligned_idx
        )
        
        # Store aligned data for correlation calculations
        diag.prediction_history_aligned = pred_aligned.copy()
        diag.actual_history_aligned = actual_aligned.copy()
        
        # Update rolling Sharpe
        rolling_sharpe = self._calculate_rolling_sharpe(returns)
        diag.rolling_sharpe = rolling_sharpe
        
        # Update recent performance (last window Sharpe)
        if len(rolling_sharpe) > 0:
            diag.recent_performance = rolling_sharpe.iloc[-1]
        
        # Update other metrics
        if len(diag.actual_history) >= self.min_active_periods:
            diag.max_dd = max_drawdown(returns)
            diag.hit_rate_value = hit_rate(diag.prediction_history.values, diag.actual_history.values)
        
        # Update last active timestamp
        diag.last_active = datetime.now()
        
        # Update regime correlation if regime provided
        if current_regime is not None:
            self._update_regime_correlation(model_id, returns, current_regime)
    
    def _calculate_rolling_sharpe(self, returns: pd.Series) -> pd.Series:
        """Calculate rolling Sharpe ratio.
        
        Args:
            returns: Return series
            
        Returns:
            Series of rolling Sharpe ratios
        """
        if len(returns) < self.lookback_window:
            return pd.Series(dtype=float)
        
        rolling_sharpe = returns.rolling(window=self.lookback_window).apply(
            lambda x: sharpe_ratio(x.values) if len(x) >= self.lookback_window else np.nan,
            raw=False
        )
        
        return rolling_sharpe.dropna()
    
    def _update_regime_correlation(
        self,
        model_id: str,
        returns: pd.Series,
        current_regime: str,
    ) -> None:
        """Update regime correlation for a model.
        
        Compute model performance correlation with each regime proxy.
        Tracks rolling Sharpe ratio during each regime type.
        
        Args:
            model_id: Model identifier
            returns: Return series (strategy returns from predictions)
            current_regime: Current regime label
        """
        diag = self.models[model_id]
        
        # Compute correlation: track Sharpe ratio during this regime
        # This measures how well the model performs in this specific regime
        if len(returns) >= self.min_active_periods:
            regime_sharpe = sharpe_ratio(returns.values)
            
            # Update or initialize regime correlation
            # Store as list of recent Sharpe ratios per regime
            if current_regime not in diag.regime_correlation:
                diag.regime_correlation[current_regime] = []
            
            # Keep recent regime performance (last 10 observations per regime)
            # This allows tracking regime-specific performance over time
            diag.regime_correlation[current_regime].append(regime_sharpe)
            if len(diag.regime_correlation[current_regime]) > 10:
                diag.regime_correlation[current_regime] = diag.regime_correlation[current_regime][-10:]
    
    def calculate_adaptive_weights(
        self,
        current_regime: Optional[str] = None,
        regime_boost: float = 0.2,
    ) -> Dict[str, float]:
        """Calculate adaptive weights for all registered models.
        
        Args:
            current_regime: Current market regime (optional)
            regime_boost: Boost factor for regime correlation (default 0.2)
            
        Returns:
            Dictionary mapping model_id to weight
        """
        # Filter active models
        active_models = {}
        now = datetime.now()
        
        for model_id, diag in self.models.items():
            # Check if model has sufficient data
            if len(diag.prediction_history) < self.min_active_periods:
                continue
            
            # Check recency (optional: could require models to be updated recently)
            # For now, we just require minimum periods
            
            active_models[model_id] = diag
        
        if len(active_models) == 0:
            return {}
        
        # Calculate base weights from recent Sharpe ratios with recency decay
        # Formula: base_weight = recent_sharpe * recency_decay_factor
        # Recency decay: exponential decay based on days since model last updated
        base_weights = {}
        for model_id, diag in active_models.items():
            recent_sharpe = diag.recent_performance
            
            # Apply recency decay: decay_factor ** days_inactive
            # Models that were updated more recently get higher weights
            recency_decay = 1.0
            if diag.last_active is not None:
                days_inactive = (now - diag.last_active).days
                recency_decay = self.sharpe_decay_factor ** days_inactive
            
            # Base weight = recent Sharpe * recency decay
            # Normalize Sharpe to positive range: add offset to handle negative Sharpe
            # Use softmax-like transformation: exp(Sharpe) for positive scaling
            # For simplicity, shift Sharpe by +2 to move most values to positive range
            sharpe_offset = recent_sharpe + 2.0  # Shift [-inf, inf] to roughly [0, inf]
            base_weights[model_id] = max(0.0, sharpe_offset * recency_decay)
        
        # Apply regime boost: increase weight for models correlated with current regime
        # Formula: regime_weight = base_weight * (1 + regime_boost * regime_performance)
        regime_weights = base_weights.copy()
        if current_regime is not None:
            for model_id, diag in active_models.items():
                if current_regime in diag.regime_correlation:
                    # Average Sharpe ratio during this regime
                    regime_performance = np.mean(diag.regime_correlation[current_regime])
                    
                    # Boost models that perform well in current regime
                    # regime_boost controls the strength of the boost (default 0.2 = 20% boost)
                    boost = 1.0 + regime_boost * max(0.0, (regime_performance + 1.0))  # Add 1.0 to ensure positive
                    regime_weights[model_id] *= boost
        
        # Normalize with minimum floor
        total_weight = sum(regime_weights.values())
        if total_weight == 0:
            # Equal weights if all are zero
            n_active = len(active_models)
            return {model_id: 1.0 / n_active for model_id in active_models.keys()}
        
        # Normalize
        normalized = {model_id: w / total_weight for model_id, w in regime_weights.items()}
        
        # Apply minimum floor with simple redistribution
        n_active = len(active_models)
        min_weight = self.min_weight_floor
        
        # Find weights below floor
        below_floor = {model_id: w for model_id, w in normalized.items() if w < min_weight}
        above_floor = {model_id: w for model_id, w in normalized.items() if w >= min_weight}
        
        if len(below_floor) > 0:
            # Calculate how much weight needs to be added
            deficit = sum(min_weight - w for w in below_floor.values())
            
            # If we have weights above floor, redistribute
            if len(above_floor) > 0 and sum(above_floor.values()) > deficit:
                # Scale down above-floor weights to free up weight
                total_above = sum(above_floor.values())
                scale_factor = (total_above - deficit) / total_above
                
                for model_id in normalized:
                    if model_id in below_floor:
                        normalized[model_id] = min_weight
                    else:
                        normalized[model_id] = normalized[model_id] * scale_factor
            else:
                # Not enough above-floor weight, just set all to floor
                normalized = {model_id: min_weight for model_id in normalized.keys()}
        
        # Final normalization to ensure sum = 1.0
        total = sum(normalized.values())
        if total > 0:
            normalized = {model_id: w / total for model_id, w in normalized.items()}
        
        return normalized
    
    def get_model_stats(self) -> pd.DataFrame:
        """Get statistics for all registered models.
        
        Returns:
            DataFrame with model statistics
        """
        stats = []
        for model_id, diag in self.models.items():
            stats.append({
                "model_id": model_id,
                "recent_sharpe": diag.recent_performance,
                "max_drawdown": diag.max_dd,
                "hit_rate": diag.hit_rate_value,
                "n_periods": len(diag.prediction_history),
                "last_active": diag.last_active,
            })
        
        return pd.DataFrame(stats)

