import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema

import itertools
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates

from utils.trend_line_optimizer import *
from utils.levels import *


class TrendLine:
    def __init__(self, comb: np.array):
        self.x = comb[:, 0].reshape(-1, 1)
        self.y = comb[:, 1]
        reg = LinearRegression().fit(self.x, self.y)
        self.y_pred = reg.predict(self.x)
        self.r2 = reg.score(self.x, self.y)
        self.std = np.std(self.x)

        # Base power of each trendline off of the distance of points apart, r2 value, and number of points
        self.power = self.r2 * 100 * len(self.x) + self.std


class Asset:
    def __init__(self, historical_data: pd.DataFrame, fractal=False):
        self.df = historical_data  # Store the historical data as a class attribute
        if fractal == True:
            self.supports, self.resistances = getFractalLevels(self.df)
        else:
            self.supports, self.resistances = find_smoothed_extrema(self.df)
        self.reversal_indexes = self.get_reversal_indexes()

    def get_reversal_indexes(self, window_size=30, min_distance=30):
        # Step 1: Calculate the Simple Moving Average (SMA)
        sma = self.df['Close'].rolling(window_size).mean()

        # Step 2: Find maxima and minima of the SMA
        sma_maxima, sma_minima = self.find_extrema(sma)

        # Step 3: Find maxima and minima of the close prices
        close_maxima, close_minima = self.find_extrema(self.df['Close'])

        # Combine all extrema points from SMA
        sma_reversal_indexes = np.concatenate((sma_maxima, sma_minima))

        # Remove negative indexes
        sma_reversal_indexes = sma_reversal_indexes[sma_reversal_indexes >= 0]

        # Sort the reversal indexes
        sma_reversal_indexes = np.sort(sma_reversal_indexes)

        # Helper function to find the nearest two extrema and return the higher or lower
        def find_nearest(index, extrema_array, is_max=True):
            if len(extrema_array) == 0:
                return None

            # Get the indices of the two nearest points by sorting based on absolute difference
            nearest_two_indices = np.argsort(np.abs(extrema_array - index))[:2]
            nearest_two = extrema_array[nearest_two_indices]

            # For maxima, return the higher of the two points
            if is_max:
                return np.max(nearest_two)
            # For minima, return the lower of the two points
            else:
                return np.min(nearest_two)

        # Step 4: Align SMA maxima to nearest close maxima, and minima to nearest close minima
        aligned_indexes = []
        for idx in sma_reversal_indexes:
            if idx in sma_maxima:
                # Align to the nearest maxima, find the higher point
                nearest_max = find_nearest(idx, close_maxima, is_max=True)
                if nearest_max is not None:
                    aligned_indexes.append(nearest_max)
            elif idx in sma_minima:
                # Align to the nearest minima, find the lower point
                nearest_min = find_nearest(idx, close_minima, is_max=False)
                if nearest_min is not None:
                    aligned_indexes.append(nearest_min)

        # Check if aligned_indexes is empty before proceeding
        if len(aligned_indexes) == 0:
            # Return an empty array if no reversal points are found
            return np.array([])

        # Step 5: Remove duplicate and very close reversal points
        aligned_indexes = np.unique(
            np.array(aligned_indexes))  # Remove duplicates
        filtered_indexes = [aligned_indexes[0]]  # Start with the first index

        for idx in aligned_indexes[1:]:
            if idx - filtered_indexes[-1] > min_distance:
                filtered_indexes.append(idx)

        # Step 6: Convert the result to a numpy array and store in the object
        return np.array(filtered_indexes)

    def find_extrema(self, close_df):
        #  Calculate local mins and max
        local_max = argrelextrema(
            close_df.values, np.greater, order=5)[0]
        local_min = argrelextrema(
            close_df.values, np.less, order=5)[0]
        return local_max.flatten(), local_min.flatten()

    def generate_fractal_trend_lines(self, num_points=3):
        """
        Automatically generate trend lines for the given price series, sorted by power.

        Args:
        - historical_data: pd.DataFrame of the asset data.
        - num_points: Number of points to fit the trend line.
        - log_transform: Whether to log transform the data.

        Returns:
        - Tuple: supports, resistances, sorted support lines, and sorted resistance lines.
        """

        supports, resistances = np.array(
            self.supports), np.array(self.resistances)

        support_lines = [TrendLine(np.array(comb))
                         for comb in itertools.combinations(supports, num_points)]
        resistance_lines = [TrendLine(
            np.array(comb)) for comb in itertools.combinations(resistances, num_points)]

        # Sort by power
        support_lines.sort(key=lambda tl: tl.power, reverse=True)
        resistance_lines.sort(key=lambda tl: tl.power, reverse=True)

        df_copy = self.df.copy()
        df_copy['Date'] = df_copy['Date'].apply(mpl_dates.date2num)

        df_copy = df_copy.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]

        fig, ax = plt.subplots()

        candlestick_ohlc(ax, df_copy.values, width=0.6,
                         colorup='green', colordown='red', alpha=0.8)
        plt.plot(df_copy['Date'][support_lines[0].x.flatten()],
                 support_lines[0].y_pred, linestyle='--')
        plt.plot(df_copy['Date'][resistance_lines[0].x.flatten()],
                 resistance_lines[0].y_pred, linestyle='--')

        plt.plot(df_copy['Date'][supports[:, 0]], supports[:, 1], 'x')
        plt.plot(df_copy['Date'][resistances[:, 0]], resistances[:, 1], 'x')

        for reversal_index in self.reversal_indexes:
            num = df_copy['Date'][reversal_index]
            plt.axvline(x=num, color='b', linestyle='dotted')

        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.show()

    def generate_pivot_trend_lines(self):

        df_copy = self.df.copy()
        support_coefs, resist_coefs = fit_trendlines_high_low(
            df_copy['High'], df_copy['Low'], df_copy['Close'])

        df_copy['Date'] = df_copy['Date'].apply(mpl_dates.date2num)

        df_copy = df_copy.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]

        fig, ax = plt.subplots()

        candlestick_ohlc(ax, df_copy.values, width=0.6,
                         colorup='green', colordown='red', alpha=0.8)

        support_line = np.arange(len(df_copy)) * \
            support_coefs[0] + support_coefs[1]
        resist_line = np.arange(len(df_copy)) * \
            resist_coefs[0] + resist_coefs[1]
        plt.plot(df_copy['Date'], support_line,
                 label="Support Line", linestyle='--')
        plt.plot(df_copy['Date'], resist_line,
                 label="Resistance Line", linestyle='--')
        plt.legend()

        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.show()

    def plot_candle(self):

        df_copy = self.df.copy()
        df_copy['Date'] = df_copy['Date'].apply(mpl_dates.date2num)

        df_copy = df_copy.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]

        fig, ax = plt.subplots()

        candlestick_ohlc(ax, df_copy.values, width=0.6,
                         colorup='green', colordown='red', alpha=0.8)
        for reversal_index in self.reversal_indexes:
            num = df_copy['Date'][reversal_index]
            plt.axvline(x=num, color='b', linestyle='dotted')

        date_format = mpl_dates.DateFormatter('%d %b %Y')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()
        fig.show()
