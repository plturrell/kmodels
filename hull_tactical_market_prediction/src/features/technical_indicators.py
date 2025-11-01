"""Technical indicators for financial time-series feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        period: RSI period (default: 14)
    
    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        DataFrame with MACD, signal, and histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram,
    })


def compute_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Compute Bollinger Bands.
    
    Args:
        prices: Price series
        period: Moving average period
        num_std: Number of standard deviations
    
    Returns:
        DataFrame with upper, middle, and lower bands
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # Compute bandwidth and %B
    bandwidth = (upper - lower) / middle
    percent_b = (prices - lower) / (upper - lower)
    
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_bandwidth': bandwidth,
        'bb_percent_b': percent_b,
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
    
    Returns:
        ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def compute_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Compute momentum indicator.
    
    Args:
        prices: Price series
        period: Lookback period
    
    Returns:
        Momentum values
    """
    return prices.diff(period)


def compute_roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """Compute Rate of Change (ROC).
    
    Args:
        prices: Price series
        period: Lookback period
    
    Returns:
        ROC values (percentage)
    """
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                       k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Compute Stochastic Oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        d_period: %D period
    
    Returns:
        DataFrame with %K and %D
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    
    return pd.DataFrame({
        'stoch_k': k,
        'stoch_d': d,
    })


def add_technical_indicators(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """Add all technical indicators to dataframe.
    
    Args:
        df: DataFrame with price data
        price_col: Name of price column
    
    Returns:
        DataFrame with added technical indicators
    """
    result = df.copy()
    prices = df[price_col]
    
    # RSI
    result['rsi_14'] = compute_rsi(prices, period=14)
    
    # MACD
    macd_df = compute_macd(prices)
    result = pd.concat([result, macd_df], axis=1)
    
    # Bollinger Bands
    bb_df = compute_bollinger_bands(prices)
    result = pd.concat([result, bb_df], axis=1)
    
    # Momentum and ROC
    result['momentum_10'] = compute_momentum(prices, period=10)
    result['roc_10'] = compute_roc(prices, period=10)
    
    # If high/low available, add ATR and Stochastic
    if 'high' in df.columns and 'low' in df.columns:
        result['atr_14'] = compute_atr(df['high'], df['low'], prices, period=14)
        stoch_df = compute_stochastic(df['high'], df['low'], prices)
        result = pd.concat([result, stoch_df], axis=1)
    
    return result


__all__ = [
    "compute_rsi",
    "compute_macd",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_momentum",
    "compute_roc",
    "compute_stochastic",
    "add_technical_indicators",
]

