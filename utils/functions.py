import pandas as pd
import numpy as np
from utils.modules import AssetLevels, TrendLine
import itertools


def simulate_gbm(S0, mu, sigma, T, dt, N):
    """
    Simulate price series using the Geometric Brownian Motion (GBM) model.

    Args:
    S0 (float): Initial asset price.
    mu (float): Drift (expected return).
    sigma (float): Volatility.
    T (float): Total time in years.
    dt (float): Time step size.
    N (int): Number of simulations.

    Returns:
    np.ndarray: Simulated price series.
    """
    np.random.seed(42)  # Fixing seed for reproducibility
    timesteps = int(T / dt)
    time = np.linspace(0, T, timesteps)

    # Generate random noise for each simulation
    W = np.random.standard_normal((timesteps, N))

    # Calculate the GBM process
    S = np.zeros_like(W)
    S[0] = S0
    for t in range(1, timesteps):
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) *
                               dt + sigma * np.sqrt(dt) * W[t])

    return S, time


def generate_fractal_trend_lines(historical_data: pd.DataFrame, num_points=4, log_transform=False):
    """
    Automatically generate a list of trend lines for the given price series (sorted by power)

    Args:
    historical data (pd.DataFrame): The price series with all information
    num_points (int): Number of points to consider in a trend line
    log_transform (bool): Whether to apply log transformation to the price series.

    Returns:
    tuple: A tuple containing the supports, resistances, and sorted lists of upper and lower trend lines.
    """
    if log_transform:
        historical_data['Open'] = np.log(historical_data['Open'])

    # Identify support and resistance points
    asset = AssetLevels(historical_data)
    supports, resistances = asset.getLevels()

    # Fit linear regressions to supports and resistances
    support_lines = []
    resistance_lines = []
    for comb in itertools.combinations(supports, num_points):
        comb = np.array(comb)
        tl = TrendLine(comb)
        support_lines.append(tl)

    for comb in itertools.combinations(resistances, num_points):
        comb = np.array(comb)
        tl = TrendLine(comb)
        resistance_lines.append(tl)
    support_lines.sort(key=lambda x: x.power, reverse=True)
    resistance_lines.sort(key=lambda x: x.power, reverse=True)

    return supports, resistances, support_lines, resistance_lines
