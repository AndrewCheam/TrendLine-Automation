import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema

import itertools
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates

from utils.trend_line_optimizer import *
from utils.fractal_levels import getFractalLevels


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
    def __init__(self, historical_data: pd.DataFrame):
        self.df = historical_data  # Store the historical data as a class attribute
        sma30 = self.df['Close'].rolling(30).mean()
        max, min = self.find_extrema(sma30)
        self.reversal_index = np.concatenate((max, min))
        self.supports, self.resistances = getFractalLevels(self.df)

    def find_extrema(self, close_df):
        #  Calculate local mins and max
        smoothed_local_max = argrelextrema(
            close_df.values, np.greater, order=5)[0]
        smoothed_local_min = argrelextrema(
            close_df.values, np.less, order=5)[0]
        return smoothed_local_max.flatten(), smoothed_local_min.flatten()

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
