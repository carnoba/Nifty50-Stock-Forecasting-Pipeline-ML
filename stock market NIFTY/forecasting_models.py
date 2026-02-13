import pandas as pd
import numpy as np
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)

class ForecastingModels:
    def __init__(self, df):
        self.df = df
        self.arima_model = None
        self.prophet_model = None

    def calculate_metrics(self, y_true, y_pred):
        """Calculates RMSE, MAE, and MAPE."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        mean_price = np.mean(y_true)
        rmse_pct = (rmse / mean_price) * 100
        
        logger.info(f"RMSE: {rmse:.2f} ({rmse_pct:.2f}% of mean)")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"MAPE: {mape:.2%}")
        
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "RMSE_PCT": rmse_pct}

    def train_arima(self, target_col='close', test_size=0.2):
        """Trains ARIMA model using auto_arima."""
        logger.info("Training ARIMA Model...")
        train_len = int(len(self.df) * (1 - test_size))
        train_data = self.df[target_col].iloc[:train_len]
        test_data = self.df[target_col].iloc[train_len:]
        
        # auto_arima with stepwise and information criterion
        model = auto_arima(train_data, 
                          seasonal=True, m=5, # Weekly seasonality for daily market data
                          stepwise=True, 
                          suppress_warnings=True, 
                          error_action="ignore", 
                          max_p=5, max_q=5,
                          trace=False)
        
        logger.info(f"Optimal ARIMA Order: {model.order}")
        
        forecast, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True)
        
        metrics = self.calculate_metrics(test_data, forecast)
        
        return {
            "forecast": pd.Series(forecast, index=test_data.index),
            "conf_int": conf_int,
            "metrics": metrics,
            "test_data": test_data,
            "model": model
        }

    def train_prophet(self, target_col='close', regressors=['sma20', 'RSI14', 'macd1226'], test_size=0.2):
        """Trains Prophet model with extra regressors."""
        logger.info("Training Prophet Model...")
        
        # Prepare data for Prophet
        prophet_df = self.df.reset_index().rename(columns={'index': 'ds', target_col: 'y'})
        
        train_len = int(len(prophet_df) * (1 - test_size))
        train_data = prophet_df.iloc[:train_len]
        test_data = prophet_df.iloc[train_len:]
        
        model = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False)
        
        for reg in regressors:
            if reg in self.df.columns:
                model.add_regressor(reg)
            else:
                logger.warning(f"Regressor {reg} not found in dataframe.")

        model.fit(train_data)
        
        # Forecast
        forecast_output = model.predict(test_data)
        
        y_pred = forecast_output['yhat'].values
        y_true = test_data['y'].values
        
        metrics = self.calculate_metrics(y_true, y_pred)
        
        return {
            "forecast": pd.Series(y_pred, index=test_data['ds']),
            "conf_int": forecast_output[['yhat_lower', 'yhat_upper']],
            "metrics": metrics,
            "test_data": pd.Series(y_true, index=test_data['ds']),
            "model": model
        }

    def cross_validate_arima(self, target_col='close', n_splits=5):
        """Cross-validation for ARIMA using TimeSeriesSplit."""
        logger.info(f"Performing {n_splits}-fold TimeSeries Cross-Validation for ARIMA...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        data = self.df[target_col]
        all_rmse = []

        for train_index, test_index in tscv.split(data):
            train, test = data.iloc[train_index], data.iloc[test_index]
            model = auto_arima(train, stepwise=True, suppress_warnings=True, error_action="ignore")
            forecast = model.predict(n_periods=len(test))
            all_rmse.append(np.sqrt(mean_squared_error(test, forecast)))
            
        avg_rmse = np.mean(all_rmse)
        logger.info(f"Average CV RMSE for ARIMA: {avg_rmse:.2f}")
        return avg_rmse
