import pandas as pd
import numpy as np
from utils.modules import Asset, TrendLine


def simulate_gbm_ohlc(S0, mu, sigma, T, dt, N, start_date='2023-01-01'):
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
    num_steps = int(T / dt)

    # Generate the date range starting from the specified start date
    dates = pd.date_range(start=start_date, periods=num_steps,
                          freq='B')  # 'B' for business days

    # Initialize matrices for open, close, high, and low prices
    open_prices = np.zeros((N, num_steps))
    open_prices[:, 0] = S0
    close_prices = np.zeros((N, num_steps + 1))
    high_prices = np.zeros((N, num_steps))
    low_prices = np.zeros((N, num_steps))

    # Simulate close prices using GBM
    close_prices[:, 0] = S0
    for i in range(1, num_steps + 1):
        dt_sqrt = np.sqrt(dt)
        z = np.random.standard_normal(N)  # Random shocks
        close_prices[:, i] = close_prices[:, i-1] * \
            np.exp((mu - 0.5 * sigma**2) * dt + sigma * dt_sqrt * z)

    # Simulate open prices as a random jump from previous day's close
    for i in range(1, num_steps):
        # Small Gaussian noise for overnight jumps
        overnight_jump = np.random.normal(1, 0.001, N)
        open_prices[:, i] = close_prices[:, i-1] * overnight_jump

    # Simulate high and low prices as deviations from open and close
    for i in range(num_steps):
        # Simulate daily range as a percentage of the day's prices
        daily_volatility = np.random.normal(0, 0.005, N)
        high_prices[:, i] = np.maximum(
            open_prices[:, i], close_prices[:, i+1]) * (1 + np.abs(daily_volatility))
        low_prices[:, i] = np.minimum(
            open_prices[:, i], close_prices[:, i+1]) * (1 - np.abs(daily_volatility))

    # Create a DataFrame with the Open, High, Low, and Close prices for each simulated day
    data = []
    for i in range(N):
        df = pd.DataFrame({
            'Open': open_prices[i],
            'High': high_prices[i],
            'Low': low_prices[i],
            # skip initial close price since it's used to start the simulation
            'Close': close_prices[i, 1:],
            'Date': dates
        })
        data.append(df)

    return data  # Return a list of DataFrames (one per simulated path)
