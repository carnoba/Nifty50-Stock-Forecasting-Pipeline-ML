import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads the Nifty 50 dataset and parses dates."""
        logger.info(f"Loading data from {self.file_path}")
        self.df = pd.read_csv(self.file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date')
        self.df.set_index('Date', inplace=True)
        logger.info(f"Data loaded. Shape: {self.df.shape}")
        return self.df

    def clean_and_slice(self, years=5):
        """Initial cleaning, filling gaps and slicing last N years."""
        logger.info(f"Filling missing values and slicing last {years} years.")
        # Handle weekend/holiday gaps with forward fill
        # First reindex to daily frequency to expose gaps
        all_dates = pd.date_range(start=self.df.index.min(), end=self.df.index.max(), freq='D')
        self.df = self.df.reindex(all_dates)
        self.df = self.df.ffill()
        
        # Slicing the most recent N years
        latest_date = self.df.index.max()
        start_date = latest_date - pd.DateOffset(years=years)
        self.df = self.df[self.df.index >= start_date]
        
        logger.info(f"Sliced data from {self.df.index.min()} to {self.df.index.max()}. New shape: {self.df.shape}")
        return self.df

    def check_stationarity(self, series_name='close'):
        """Performs ADF test and applies differencing if necessary."""
        series = self.df[series_name]
        result = adfuller(series.dropna())
        logger.info(f"ADF Statistic: {result[0]}")
        logger.info(f"p-value: {result[1]}")
        
        is_stationary = result[1] <= 0.05
        d = 0
        df_diff = series.copy()
        
        while not is_stationary and d < 2:
            d += 1
            df_diff = df_diff.diff().dropna()
            result = adfuller(df_diff)
            logger.info(f"After {d} order differencing, p-value: {result[1]}")
            if result[1] <= 0.05:
                is_stationary = True
        
        return is_stationary, d

    def decompose(self, series_name='close', model='additive', period=365):
        """Extracts Trend, Seasonality, and Residuals."""
        logger.info(f"Performing seasonal decomposition ({model} model).")
        result = seasonal_decompose(self.df[series_name], model=model, period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        result.observed.plot(ax=axes[0], title='Observed')
        result.trend.plot(ax=axes[1], title='Trend')
        result.seasonal.plot(ax=axes[2], title='Seasonality')
        result.resid.plot(ax=axes[3], title='Residuals')
        
        plt.tight_layout()
        plot_path = 'decomposition_plot.png'
        plt.savefig(plot_path)
        logger.info(f"Decomposition plot saved to {plot_path}")
        return result

if __name__ == "__main__":
    # Test block
    processor = DataProcessor('Nifty_50_with_indicators_.csv')
    df = processor.load_data()
    df = processor.clean_and_slice(years=5)
    stationary, d_order = processor.check_stationarity()
    processor.decompose()
