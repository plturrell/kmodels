"""Meta-feature engineering for stacking ensemble."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .adaptive_ensemble import AdaptiveEnsemble
from .regime_detection import RegimeDetector


class MetaFeatureBuilder:
    """Build enhanced meta-features for stacking."""
    
    def __init__(
        self,
        adaptive_ensemble: Optional[AdaptiveEnsemble] = None,
        regime_detector: Optional[RegimeDetector] = None,
        rolling_window: int = 21,
    ):
        """Initialize meta-feature builder.
        
        Args:
            adaptive_ensemble: Adaptive ensemble instance for performance stats
            regime_detector: Regime detector instance
            rolling_window: Window for rolling statistics (default 21)
        """
        self.adaptive_ensemble = adaptive_ensemble
        self.regime_detector = regime_detector
        self.rolling_window = rolling_window
    
    def build_meta_features(
        self,
        base_predictions: pd.DataFrame,
        original_features: Optional[pd.DataFrame] = None,
        id_column: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Build enhanced meta-features for stacking.
        
        Args:
            base_predictions: DataFrame with base model predictions (columns = model predictions)
            original_features: Original feature dataframe (for feature statistics)
            id_column: ID column for alignment (optional)
            
        Returns:
            DataFrame with enhanced meta-features
        """
        meta_features = pd.DataFrame(index=base_predictions.index)
        
        # 1. Base predictions (directly include)
        for col in base_predictions.columns:
            meta_features[f"pred_{col}"] = base_predictions[col]
        
        # 2. Prediction statistics
        meta_features['pred_mean'] = base_predictions.mean(axis=1)
        meta_features['pred_std'] = base_predictions.std(axis=1)
        meta_features['pred_max'] = base_predictions.max(axis=1)
        meta_features['pred_min'] = base_predictions.min(axis=1)
        meta_features['pred_range'] = meta_features['pred_max'] - meta_features['pred_min']
        
        # 3. Regime indicators
        if self.regime_detector is not None and original_features is not None:
            regime_proxy = self.regime_detector.get_regime_proxy(original_features)
            # Align indices
            if len(regime_proxy) == len(meta_features):
                for col in regime_proxy.columns:
                    meta_features[f"regime_{col}"] = regime_proxy[col].values
            else:
                # Try to align by index
                aligned_proxy = regime_proxy.reindex(meta_features.index, fill_value=0.0)
                for col in regime_proxy.columns:
                    meta_features[f"regime_{col}"] = aligned_proxy[col].values
        
        # 4. Performance stats per model (from AdaptiveEnsemble)
        if self.adaptive_ensemble is not None:
            model_stats = self.adaptive_ensemble.get_model_stats()
            if len(model_stats) > 0:
                for _, row in model_stats.iterrows():
                    model_id = row['model_id']
                    if model_id in base_predictions.columns:
                        meta_features[f"stats_{model_id}_sharpe"] = row['recent_sharpe']
                        meta_features[f"stats_{model_id}_maxdd"] = row['max_drawdown']
                        meta_features[f"stats_{model_id}_hitrate"] = row['hit_rate']
                    else:
                        # Broadcast stats if model not in predictions
                        meta_features[f"stats_{model_id}_sharpe"] = row['recent_sharpe']
                        meta_features[f"stats_{model_id}_maxdd"] = row['max_drawdown']
                        meta_features[f"stats_{model_id}_hitrate"] = row['hit_rate']
        
        # 5. Feature statistics from original features
        # Volatility features: V1-V13 for volatility
        # Market features: M1-M18 for market/momentum
        # Economic indicators: E1-E20 for economic
        if original_features is not None:
            # Volatility features (V1-V13): Rolling std of V* columns
            vol_cols = [col for col in original_features.columns 
                       if col.startswith('V') and (len(col) < 3 or col[1:].isdigit())]
            if len(vol_cols) > 0:
                vol_df = original_features[vol_cols]
                meta_features['feat_vol_mean'] = vol_df.mean(axis=1)
                meta_features['feat_vol_std'] = vol_df.std(axis=1)
                meta_features['feat_vol_max'] = vol_df.max(axis=1)
                meta_features['feat_vol_min'] = vol_df.min(axis=1)
                
                # Rolling volatility statistics (rolling std as specified)
                vol_series = vol_df.mean(axis=1)  # Aggregate first, then roll
                meta_features['feat_vol_rolling_std'] = vol_series.rolling(
                    window=self.rolling_window, min_periods=1
                ).std().fillna(0.0)
                meta_features['feat_vol_rolling_mean'] = vol_series.rolling(
                    window=self.rolling_window, min_periods=1
                ).mean().fillna(0.0)
            
            # Market features (M1-M18): Rolling mean of M* momentum columns
            market_cols = [col for col in original_features.columns 
                          if col.startswith('M') and (len(col) < 3 or col[1:].isdigit())]
            if len(market_cols) > 0:
                market_df = original_features[market_cols]
                meta_features['feat_market_mean'] = market_df.mean(axis=1)
                meta_features['feat_market_std'] = market_df.std(axis=1)
                
                # Rolling momentum (rolling mean as specified)
                market_series = market_df.mean(axis=1)  # Aggregate first
                meta_features['feat_market_rolling_mean'] = market_series.rolling(
                    window=self.rolling_window, min_periods=1
                ).mean().fillna(0.0)
                meta_features['feat_market_rolling_std'] = market_series.rolling(
                    window=self.rolling_window, min_periods=1
                ).std().fillna(0.0)
                
                # Trend strength: difference in rolling means (short vs long)
                short_window = max(5, self.rolling_window // 4)
                meta_features['feat_market_trend_strength'] = (
                    market_series.rolling(window=short_window, min_periods=1).mean() -
                    market_series.rolling(window=self.rolling_window, min_periods=1).mean()
                ).fillna(0.0)
            
            # Economic indicators (E1-E20): Recent E* feature values
            econ_cols = [col for col in original_features.columns 
                        if col.startswith('E') and (len(col) < 3 or col[1:].isdigit())]
            if len(econ_cols) > 0:
                econ_df = original_features[econ_cols]
                meta_features['feat_econ_mean'] = econ_df.mean(axis=1)
                meta_features['feat_econ_std'] = econ_df.std(axis=1)
                
                # Recent economic indicators (short rolling window)
                econ_series = econ_df.mean(axis=1)
                recent_window = min(7, self.rolling_window // 3)  # Short window for recent trends
                meta_features['feat_econ_recent_mean'] = econ_series.rolling(
                    window=recent_window, min_periods=1
                ).mean().fillna(0.0)
                meta_features['feat_econ_recent_std'] = econ_series.rolling(
                    window=recent_window, min_periods=1
                ).std().fillna(0.0)
        
        # Handle NaN values
        meta_features = meta_features.fillna(0.0)
        
        return meta_features
    
    def build_meta_features_incremental(
        self,
        base_predictions: pd.DataFrame,
        original_features: Optional[pd.DataFrame] = None,
        previous_meta_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build meta-features incrementally (for online prediction).
        
        Args:
            base_predictions: Current base model predictions
            original_features: Current original features
            previous_meta_features: Previous meta-features (for rolling stats)
            
        Returns:
            DataFrame with enhanced meta-features
        """
        # For incremental, we can reuse the main method if indices align
        # This is a placeholder for more sophisticated incremental logic
        return self.build_meta_features(base_predictions, original_features)

