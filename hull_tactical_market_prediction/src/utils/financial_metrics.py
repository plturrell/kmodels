"""Financial and risk-adjusted performance metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def sharpe_ratio(returns: np.ndarray | pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute Sharpe Ratio (risk-adjusted return).
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)
    
    Returns:
        Annualized Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)


def sortino_ratio(returns: np.ndarray | pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute Sortino Ratio (downside risk-adjusted return).
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized Sortino ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    downside_std = np.std(downside_returns)
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    """Compute maximum drawdown.
    
    Args:
        returns: Return series
    
    Returns:
        Maximum drawdown (negative value)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    
    return np.min(drawdown)


def calmar_ratio(returns: np.ndarray | pd.Series, periods_per_year: int = 252) -> float:
    """Compute Calmar Ratio (return / max drawdown).
    
    Args:
        returns: Return series
        periods_per_year: Number of periods per year
    
    Returns:
        Calmar ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    annualized_return = np.mean(returns) * periods_per_year
    mdd = max_drawdown(returns)
    
    if mdd == 0:
        return 0.0
    
    return annualized_return / abs(mdd)


def information_ratio(returns: np.ndarray | pd.Series, benchmark_returns: np.ndarray | pd.Series, 
                      periods_per_year: int = 252) -> float:
    """Compute Information Ratio (excess return / tracking error).
    
    Args:
        returns: Strategy returns
        benchmark_returns: Benchmark returns
        periods_per_year: Number of periods per year
    
    Returns:
        Annualized information ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    if isinstance(benchmark_returns, pd.Series):
        benchmark_returns = benchmark_returns.values
    
    excess_returns = returns - benchmark_returns
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / tracking_error


def hit_rate(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Compute hit rate (directional accuracy).
    
    Args:
        predictions: Predicted returns
        actuals: Actual returns
    
    Returns:
        Hit rate (0.0 to 1.0)
    """
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actuals)
    
    return np.mean(pred_direction == actual_direction)


def profit_factor(returns: np.ndarray | pd.Series) -> float:
    """Compute profit factor (gross profit / gross loss).
    
    Args:
        returns: Return series
    
    Returns:
        Profit factor
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def compute_all_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Compute all financial metrics.
    
    Args:
        predictions: Predicted returns
        actuals: Actual returns
        risk_free_rate: Risk-free rate
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary of all metrics
    """
    # Prediction error metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    # Risk-adjusted metrics (using actual returns)
    sharpe = sharpe_ratio(actuals, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(actuals, risk_free_rate, periods_per_year)
    mdd = max_drawdown(actuals)
    calmar = calmar_ratio(actuals, periods_per_year)
    
    # Directional accuracy
    hit = hit_rate(predictions, actuals)
    pf = profit_factor(actuals)
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(mdd),
        "calmar_ratio": float(calmar),
        "hit_rate": float(hit),
        "profit_factor": float(pf),
    }


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "information_ratio",
    "hit_rate",
    "profit_factor",
    "compute_all_metrics",
]

