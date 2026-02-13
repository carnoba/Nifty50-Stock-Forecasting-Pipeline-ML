import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from data_processor import DataProcessor
from forecasting_models import ForecastingModels

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dashboard(df, arima_res, prophet_res):
    """Creates a consolidated dashboard plot."""
    logger.info("Creating professional dashboard...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

    # Subplot 1: Forecast Comparison
    ax1 = fig.add_subplot(gs[0])
    # Training Data (slice last year of train for clarity)
    train_end = arima_res['test_data'].index[0]
    train_plot = df['close'].loc[:train_end].tail(250)
    ax1.plot(train_plot.index, train_plot, label='Training Data (Last 250d)', color='gray', alpha=0.6)
    
    # Test Data
    ax1.plot(arima_res['test_data'].index, arima_res['test_data'], label='Actual Price', color='white', linewidth=2)
    
    # ARIMA Forecast
    ax1.plot(arima_res['forecast'].index, arima_res['forecast'], label='ARIMA Forecast', color='cyan', linestyle='--')
    ax1.fill_between(arima_res['forecast'].index, 
                    arima_res['conf_int'][:, 0], 
                    arima_res['conf_int'][:, 1], 
                    color='cyan', alpha=0.1, label='ARIMA 95% CI')
    
    # Prophet Forecast
    ax1.plot(prophet_res['forecast'].index, prophet_res['forecast'], label='Prophet Forecast', color='orange', linestyle='--')
    ax1.fill_between(prophet_res['forecast'].index, 
                    prophet_res['conf_int']['yhat_lower'], 
                    prophet_res['conf_int']['yhat_upper'], 
                    color='orange', alpha=0.1, label='Prophet 95% CI')
    
    ax1.set_title('NIFTY 50 Index: ARIMA vs Prophet Forecast Comparison', fontsize=16)
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Residual Analysis (Histogram)
    ax2 = fig.add_subplot(gs[1])
    arima_residuals = arima_res['test_data'] - arima_res['forecast']
    prophet_residuals = prophet_res['test_data'] - prophet_res['forecast']
    
    sns.histplot(arima_residuals, kde=True, color='cyan', label='ARIMA Residuals', ax=ax2, alpha=0.5)
    sns.histplot(prophet_residuals, kde=True, color='orange', label='Prophet Residuals', ax=ax2, alpha=0.5)
    ax2.set_title('Residual Error Distribution (Check for Normalcy)', fontsize=14)
    ax2.legend()

    # Subplot 3: Residual vs Time
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(arima_residuals.index, arima_residuals, color='cyan', label='ARIMA Residuals', alpha=0.7)
    ax3.plot(prophet_residuals.index, prophet_residuals, color='orange', label='Prophet Residuals', alpha=0.7)
    ax3.axhline(0, color='red', linestyle='--')
    ax3.set_title('Residual Errors Over Time (Check for White Noise)', fontsize=14)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('nifty_forecast_dashboard.png')
    logger.info("Dashboard saved to nifty_forecast_dashboard.png")

def main():
    file_path = 'Nifty_50_with_indicators_.csv'
    
    # Step 1: Preprocessing
    processor = DataProcessor(file_path)
    df = processor.load_data()
    df = processor.clean_and_slice(years=5)
    
    stationary, d_order = processor.check_stationarity()
    logger.info(f"Stationarity Check: {'Stationary' if stationary else 'Non-Stationary'}, d={d_order}")
    
    processor.decompose()
    
    # Step 2: Modeling
    forecaster = ForecastingModels(df)
    
    # ARIMA
    arima_results = forecaster.train_arima(test_size=0.2)
    logger.info(f"Step 2: ARIMA metrics = {arima_results['metrics']}")
    
    # Prophet
    # Indicators based on CSV inspection
    indicators = ['sma20', 'RSI14', 'macd1226'] 
    prophet_results = forecaster.train_prophet(regressors=indicators, test_size=0.2)
    logger.info(f"Step 2: Prophet metrics = {prophet_results['metrics']}")
    
    # Cross-validation (Optional, takes time)
    # forecaster.cross_validate_arima()
    
    # Step 3 & 4: Visualization
    create_dashboard(df, arima_results, prophet_results)
    
    logger.info("Forecasting Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
