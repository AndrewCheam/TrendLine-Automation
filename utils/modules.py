import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


class TrendLine:
    def __init__(self, comb: np.array):
        self.x = comb[:, 0].reshape(-1, 1)
        self.y = comb[:, 1]
        reg = LinearRegression().fit(self.x, self.y)
        self.y_pred = reg.predict(self.x)
        self.r2 = reg.score(self.x, self.y)
        self.std = np.std(self.x)

        # Base power of each trendline off of the distance of points apart and r2 value, need to scale somehow
        self.power = self.r2 * 200  # + self.std


class AssetLevels:
    def __init__(self, historical_data: pd.DataFrame):
        self.df = historical_data  # Store the historical data as a class attribute
        # Average range for detecting levels
        # self.s = np.mean(self.df['High'] - self.df['Low'])/2
        self.s = 0
        self.levels = []  # All detected levels (support and resistance)
        self.supports = []  # Detected support levels
        self.resistances = []  # Detected resistance levels
        self.populateLevels()

    def isSupport(self, i):
        """
        Checks if a point is a support level (local minimum).
        """
        df = self.df
        support = df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] \
            and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
        return support

    def isResistance(self, i):
        """
        Checks if a point is a resistance level (local maximum).
        """
        df = self.df
        resistance = df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] \
            and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
        return resistance

    def isFarFromLevel(self, l):
        """
        Checks if the detected level is far enough from existing levels to be considered new.
        """
        return np.sum([abs(l - x) < self.s for x in self.levels]) == 0

    def populateLevels(self):
        """
        Detects all support and resistance levels in the data and stores them.
        """
        for i in range(2, self.df.shape[0] - 2):
            if self.isSupport(i):
                l = self.df['Low'][i]
                if self.isFarFromLevel(l):
                    self.supports.append((i, l))  # Add support to list
                    self.levels.append(l)  # Track all levels

            elif self.isResistance(i):
                l = self.df['High'][i]
                if self.isFarFromLevel(l):
                    self.resistances.append((i, l))  # Add resistance to list
                    self.levels.append(l)  # Track all levels
        return self.supports, self.resistances

    def getLevels(self):
        """
        Returns a tuple of (supports, resistances)
        """
        return np.array(self.supports), np.array(self.resistances)
