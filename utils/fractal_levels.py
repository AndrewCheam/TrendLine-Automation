import numpy as np


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
