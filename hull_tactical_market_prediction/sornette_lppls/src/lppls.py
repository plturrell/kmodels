"""Core implementation of the Log-Periodic Power Law Singularity (LPPLS) model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def lppls_model(t, tc, m, omega, A, B, C, phi):
    """Log-Periodic Power Law Singularity (LPPLS) model function."""
    dt = tc - t
    return A + (B * (dt**m)) * (1 + C * np.cos(omega * np.log(dt) + phi))


def lppls_residuals(params, t, y):
    """Residuals for the LPPLS model, used for optimization."""
    tc, m, omega, A, B, C, phi = params
    return lppls_model(t, tc, m, omega, A, B, C, phi) - y


def fit_lppls(price_series: np.ndarray):
    """Fit the LPPLS model to a given price series.

    Args:
        price_series: A numpy array of prices.

    Returns:
        A dictionary containing the fitted parameters and the optimization result,
        or None if the optimization fails.
    """
    t = np.arange(len(price_series))
    log_price = np.log(price_series)

    # Set initial parameter guesses and bounds
    # These are critical for successful optimization
    initial_guess = [
        len(t) * 1.1,  # tc (critical time)
        0.5,           # m (exponent)
        10,            # omega (log-frequency)
        np.mean(log_price), # A (intercept)
        -0.5,          # B (amplitude)
        0.5,           # C (cosine amplitude)
        0,             # phi (phase)
    ]

    bounds = (
        [len(t), 0.1, 1, -np.inf, -np.inf, -1, -np.pi], # Lower bounds
        [len(t) * 1.5, 0.9, 50, np.inf, np.inf, 1, np.pi]  # Upper bounds
    )

    try:
        result = least_squares(
            lppls_residuals,
            x0=initial_guess,
            args=(t, log_price),
            bounds=bounds,
            method="trf",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )

        if not result.success:
            return None

        fitted_params = {
            "tc": result.x[0],
            "m": result.x[1],
            "omega": result.x[2],
            "A": result.x[3],
            "B": result.x[4],
            "C": result.x[5],
            "phi": result.x[6],
        }

        return {"params": fitted_params, "result": result}

    except Exception:
        return None
