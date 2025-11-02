"""Regime detection using volatility, market, and economic features."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


class RegimeDetector:
    """Detect market regimes from feature groups."""
    
    def __init__(
        self,
        volatility_threshold: float = 0.75,  # 75th percentile as specified
        momentum_window: int = 21,
    ):
        """Initialize regime detector.
        
        Args:
            volatility_threshold: Percentile threshold for high volatility (default 0.75 = 75th percentile)
            momentum_window: Window for momentum calculation (default 21)
        """
        self.volatility_threshold = volatility_threshold
        self.momentum_window = momentum_window
        
        # Store historical thresholds for percentile-based detection
        # High volatility: above 75th percentile of V* features
        self.volatility_percentiles: Optional[pd.Series] = None
        self.volatility_threshold_value: Optional[float] = None
        self.momentum_percentiles: Optional[pd.Series] = None
        self.momentum_threshold_value: Optional[float] = None
    
    def _get_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility features (V* columns: V1-V13 for volatility).
        
        Args:
            df: Feature dataframe
            
        Returns:
            DataFrame with volatility features
        """
        # Specifically look for V1-V13 pattern, but also accept any V-prefixed columns
        vol_cols = []
        for col in df.columns:
            if col.startswith('V'):
                # Try to match V1-V13 pattern
                if len(col) >= 2:
                    try:
                        num = int(col[1:])
                        if 1 <= num <= 13:
                            vol_cols.append(col)
                        elif col not in vol_cols:  # Also include other V* columns
                            vol_cols.append(col)
                    except ValueError:
                        # Non-numeric suffix, include it
                        if col not in vol_cols:
                            vol_cols.append(col)
        
        if len(vol_cols) == 0:
            return pd.DataFrame()
        return df[vol_cols]
    
    def _get_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract market features (M* columns: M1-M18 for market/momentum).
        
        Args:
            df: Feature dataframe
            
        Returns:
            DataFrame with market features
        """
        # Specifically look for M1-M18 pattern, but also accept any M-prefixed columns
        market_cols = []
        for col in df.columns:
            if col.startswith('M'):
                # Try to match M1-M18 pattern
                if len(col) >= 2:
                    try:
                        num = int(col[1:])
                        if 1 <= num <= 18:
                            market_cols.append(col)
                        elif col not in market_cols:  # Also include other M* columns
                            market_cols.append(col)
                    except ValueError:
                        # Non-numeric suffix, include it
                        if col not in market_cols:
                            market_cols.append(col)
        
        if len(market_cols) == 0:
            return pd.DataFrame()
        return df[market_cols]
    
    def _get_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract economic features (E* columns: E1-E20 for economic).
        
        Args:
            df: Feature dataframe
            
        Returns:
            DataFrame with economic features
        """
        # Specifically look for E1-E20 pattern, but also accept any E-prefixed columns
        econ_cols = []
        for col in df.columns:
            if col.startswith('E'):
                # Try to match E1-E20 pattern
                if len(col) >= 2:
                    try:
                        num = int(col[1:])
                        if 1 <= num <= 20:
                            econ_cols.append(col)
                        elif col not in econ_cols:  # Also include other E* columns
                            econ_cols.append(col)
                    except ValueError:
                        # Non-numeric suffix, include it
                        if col not in econ_cols:
                            econ_cols.append(col)
        
        if len(econ_cols) == 0:
            return pd.DataFrame()
        return df[econ_cols]
    
    def _compute_volatility_aggregate(self, vol_df: pd.DataFrame) -> pd.Series:
        """Compute aggregate volatility measure.
        
        Args:
            vol_df: Volatility features dataframe
            
        Returns:
            Series with aggregate volatility per row
        """
        if vol_df.empty:
            return pd.Series(index=vol_df.index, dtype=float)
        
        # Use mean of absolute values as volatility aggregate
        return vol_df.abs().mean(axis=1)
    
    def _compute_momentum_strength(self, market_df: pd.DataFrame) -> pd.Series:
        """Compute momentum strength from market features.
        
        Args:
            market_df: Market features dataframe
            
        Returns:
            Series with momentum strength per row
        """
        if market_df.empty:
            return pd.Series(index=market_df.index, dtype=float)
        
        # Use rolling mean change as momentum proxy
        momentum = market_df.rolling(window=self.momentum_window, min_periods=1).mean().diff(axis=0)
        # Aggregate momentum strength
        momentum_strength = momentum.abs().mean(axis=1)
        return momentum_strength.fillna(0.0)
    
    def _compute_mean_reversion_signal(self, market_df: pd.DataFrame) -> pd.Series:
        """Compute mean reversion signal.
        
        Args:
            market_df: Market features dataframe
            
        Returns:
            Series with mean reversion signal per row
        """
        if market_df.empty:
            return pd.Series(index=market_df.index, dtype=float)
        
        # Low momentum + high variance suggests mean reversion
        momentum = self._compute_momentum_strength(market_df)
        variance = market_df.rolling(window=self.momentum_window, min_periods=1).std().mean(axis=1)
        
        # Mean reversion = high variance / (low momentum + epsilon)
        mean_reversion = variance / (momentum + 1e-6)
        return mean_reversion.fillna(0.0)
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit regime detector on historical data (calculate percentiles).
        
        High volatility: High V* features (above 75th percentile)
        Trending: Strong momentum in M* features (trend strength)
        Mean reverting: Low momentum, high mean-reversion signals
        
        Args:
            df: Historical feature dataframe
        """
        vol_df = self._get_volatility_features(df)
        market_df = self._get_market_features(df)
        
        if not vol_df.empty:
            vol_agg = self._compute_volatility_aggregate(vol_df)
            # Store 75th percentile threshold for high volatility detection
            self.volatility_threshold_value = vol_agg.quantile(self.volatility_threshold)
            self.volatility_percentiles = vol_agg  # Store full series for reference
        
        if not market_df.empty:
            momentum = self._compute_momentum_strength(market_df)
            # Store momentum thresholds for trending detection
            self.momentum_threshold_value = momentum.quantile(self.volatility_threshold)  # 75th percentile
            self.momentum_percentiles = momentum  # Store full series for reference
    
    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect regime for each row in dataframe.
        
        Args:
            df: Feature dataframe
            
        Returns:
            Series of regime labels: 'high_vol', 'trending', 'mean_reverting', or 'normal'
        """
        vol_df = self._get_volatility_features(df)
        market_df = self._get_market_features(df)
        
        n_samples = len(df)
        regimes = pd.Series(index=df.index, dtype=str, data='normal')
        
        # Detect high volatility: High V* features (above 75th percentile)
        if not vol_df.empty and self.volatility_threshold_value is not None:
            vol_agg = self._compute_volatility_aggregate(vol_df)
            high_vol_mask = vol_agg >= self.volatility_threshold_value
            regimes[high_vol_mask] = 'high_vol'
        
        # Detect trending vs mean reverting (only if not high vol)
        # Trending: Strong momentum in M* features (trend strength)
        # Mean reverting: Low momentum, high mean-reversion signals
        if not market_df.empty:
            momentum = self._compute_momentum_strength(market_df)
            mean_reversion = self._compute_mean_reversion_signal(market_df)
            
            # Trending: high momentum (above threshold), low mean reversion
            if self.momentum_threshold_value is not None:
                momentum_threshold = self.momentum_threshold_value  # 75th percentile
            else:
                # Fallback if not fitted
                momentum_threshold = momentum.quantile(0.75)
            
            # Mean reversion threshold (60th percentile as reasonable default)
            mean_rev_threshold = mean_reversion.quantile(0.6)
            
            # Trending: high momentum AND low mean reversion
            trending_mask = (momentum >= momentum_threshold) & (mean_reversion < mean_rev_threshold)
            # Mean reverting: low momentum AND high mean reversion
            mean_rev_mask = (momentum < momentum_threshold) & (mean_reversion >= mean_rev_threshold)
            
            # Only override if not already high_vol
            normal_mask = regimes == 'normal'
            regimes[normal_mask & trending_mask] = 'trending'
            regimes[normal_mask & mean_rev_mask] = 'mean_reverting'
        
        return regimes
    
    def get_regime_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get normalized regime indicators as proxy values.
        
        Args:
            df: Feature dataframe
            
        Returns:
            DataFrame with regime proxy columns: high_vol_proxy, trending_proxy, mean_reverting_proxy
        """
        vol_df = self._get_volatility_features(df)
        market_df = self._get_market_features(df)
        
        n_samples = len(df)
        proxies = pd.DataFrame(index=df.index)
        
        # High volatility proxy
        if not vol_df.empty:
            vol_agg = self._compute_volatility_aggregate(vol_df)
            if self.volatility_percentiles is not None:
                proxies['high_vol_proxy'] = (vol_agg / (self.volatility_percentiles + 1e-6)).clip(0, 2)
            else:
                proxies['high_vol_proxy'] = (vol_agg / (vol_agg.max() + 1e-6)).clip(0, 1)
        else:
            proxies['high_vol_proxy'] = 0.0
        
        # Trending proxy
        if not market_df.empty:
            momentum = self._compute_momentum_strength(market_df)
            if self.momentum_percentiles is not None:
                proxies['trending_proxy'] = (momentum / (self.momentum_percentiles + 1e-6)).clip(0, 2)
            else:
                proxies['trending_proxy'] = (momentum / (momentum.max() + 1e-6)).clip(0, 1)
            
            mean_reversion = self._compute_mean_reversion_signal(market_df)
            mean_rev_max = mean_reversion.max() if len(mean_reversion) > 0 else 1.0
            proxies['mean_reverting_proxy'] = (mean_reversion / (mean_rev_max + 1e-6)).clip(0, 1)
        else:
            proxies['trending_proxy'] = 0.0
            proxies['mean_reverting_proxy'] = 0.0
        
        # Normal proxy (inverse of others)
        proxies['normal_proxy'] = 1.0 - (
            proxies['high_vol_proxy'].clip(0, 1) * 0.4 +
            proxies['trending_proxy'].clip(0, 1) * 0.3 +
            proxies['mean_reverting_proxy'].clip(0, 1) * 0.3
        )
        proxies['normal_proxy'] = proxies['normal_proxy'].clip(0, 1)
        
        return proxies

