from scipy.signal import argrelextrema
from statsmodels.nonparametric.kernel_regression import KernelReg
import numpy as np
import pandas as pd


def isSupport(df, i):
    """
    Checks if a point is a support level (local minimum).
    """
    support = df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] \
        and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
    return support


def isResistance(df, i):
    """
    Checks if a point is a resistance level (local maximum).
    """
    resistance = df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] \
        and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
    return resistance


def isFarFromLevel(s, l, levels):
    """
    Checks if the detected level is far enough from existing levels to be considered new.
    """
    return np.sum([abs(l - x) < s for x in levels]) == 0


def getFractalLevels(df):
    """
    Detects all support and resistance levels in the data and stores them.
    """
    supports = []
    resistances = []
    levels = []
    s = 0
    # s = np.mean(df['High'] - df['Low'])/2
    for i in range(2, df.shape[0] - 2):
        if isSupport(df, i):
            l = df['Low'][i]
            if isFarFromLevel(s, l, levels):
                supports.append((i, l))  # Add support to list
                levels.append(l)  # Track all levels

        elif isResistance(df, i):
            l = df['High'][i]
            if isFarFromLevel(s, l, levels):
                resistances.append((i, l))  # Add resistance to list
                levels.append(l)  # Track all levels
    return supports, resistances


def get_smoothed_curve(close_df, bandwidth):
    #  Use Kernel Regression to create a fitted curve
    kernel_regression = KernelReg(
        [close_df.values], [close_df.index], var_type='c', bw=bandwidth)
    regression_result = kernel_regression.fit([close_df.index])
    # Get smoothed close prices
    smoothed_close_df = pd.Series(
        data=regression_result[0], index=close_df.index)
    return smoothed_close_df


def find_smoothed_extrema(df):
    #  Calculate local mins and max
    close_df = df['Close']
    smoothed_close_df = get_smoothed_curve(close_df, bandwidth='cv_ls')
    smoothed_local_max = argrelextrema(
        smoothed_close_df.values, np.greater, order=5)[0]
    smoothed_local_min = argrelextrema(
        smoothed_close_df.values, np.less, order=5)[0]

    smoothed_local_max = np.concatenate(
        (smoothed_local_max.reshape(-1, 1), close_df.iloc[smoothed_local_max].values.reshape(-1, 1)), axis=1)
    smoothed_local_min = np.concatenate(
        (smoothed_local_min.reshape(-1, 1), close_df.iloc[smoothed_local_min].values.reshape(-1, 1)), axis=1)

    return smoothed_local_max, smoothed_local_min
