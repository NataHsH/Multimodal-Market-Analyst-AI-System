# forecast_arima.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def forecast_arima(df, forecast_period=90):
    """
    Forecast stock prices using ARIMA model.
    Parameters:
        df (pd.DataFrame): DataFrame with columns 'ds' (datetime) and 'y' (values)
        forecast_period (int): Number of days to forecast
    Returns:
        pd.DataFrame: Forecasted values with columns 'ds', 'forecast', 'lower_bound', 'upper_bound'
    """
    # Set datetime column as index
    ts = df.set_index('ds')['y']

    # Fit ARIMA model with example order (p=5, d=1, q=0), tune if necessary
    model = ARIMA(ts, order=(5,1,0))
    model_fit = model.fit()

    # Get forecast for specified period with 95% confidence interval
    forecast_result = model_fit.get_forecast(steps=forecast_period)
    forecast_df = forecast_result.summary_frame(alpha=0.05)

    # Create dates for forecast period
    forecast_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_period)

    # Prepare forecast DataFrame
    forecast = pd.DataFrame({
        'ds': forecast_dates,
        'forecast': forecast_df['mean'].values,
        'lower_bound': forecast_df['mean_ci_lower'].values,
        'upper_bound': forecast_df['mean_ci_upper'].values
    })

    return forecast

def plot_forecast_arima(df, forecast):
    """
    Plot historical stock prices and ARIMA forecast with confidence intervals.
    Parameters:
        df (pd.DataFrame): Historical data with 'ds' and 'y' columns
        forecast (pd.DataFrame): Forecast data with 'ds', 'forecast', 'lower_bound', 'upper_bound'
    """
    plt.figure(figsize=(12,6))
    plt.plot(df['ds'], df['y'], label='Historical')
    plt.plot(forecast['ds'], forecast['forecast'], label='Forecast', color='orange')
    plt.fill_between(forecast['ds'], forecast['lower_bound'], forecast['upper_bound'], color='orange', alpha=0.3, label='95% CI')
    plt.legend()
    plt.title("ARIMA Forecast with 95% Confidence Interval")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Example data: replace with your real data
    data = {
        'ds': pd.date_range(start='2020-01-01', periods=100),
        'y': pd.Series(range(100)) + 10 * pd.Series(pd.np.random.randn(100)).cumsum()
    }
    df = pd.DataFrame(data)

    # Forecast next 90 days
    forecast_df = forecast_arima(df, forecast_period=90)

    # Plot results
    plot_forecast_arima(df, forecast_df)
