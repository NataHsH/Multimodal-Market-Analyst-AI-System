import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def forecast_stock_price(df):
    """
    Train Prophet model on historical data and forecast next 90 days.
    df: pandas DataFrame with columns 'ds' (datetime) and 'y' (value)
    Returns forecast DataFrame with predicted values and confidence intervals.
    """
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def plot_forecast(df, forecast):
    """
    Plot historical data and forecast with confidence intervals.
    """
    plt.figure(figsize=(10,6))
    plt.plot(df['ds'], df['y'], label='Historical')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3, label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Forecast')
    plt.legend()
    plt.show()

def generate_report(forecast):
    """
    Generate a simple text report based on forecast data.
    """
    last_date = forecast['ds'].iloc[-1].date()
    last_pred = forecast['yhat'].iloc[-1]
    lower = forecast['yhat_lower'].iloc[-1]
    upper = forecast['yhat_upper'].iloc[-1]

    report = (f"Forecast for {last_date}:\n"
              f"Expected stock price: {last_pred:.2f}\n"
              f"Confidence interval: [{lower:.2f}, {upper:.2f}]\n")

    # Add simple trend comment
    if last_pred > forecast['yhat'].iloc[-91]:
        report += "Trend is upward compared to 3 months ago.\n"
    else:
        report += "Trend is downward compared to 3 months ago.\n"

    return report

if __name__ == "__main__":
    # Example data
    data = {
        'ds': pd.date_range(start='2024-01-01', periods=180, freq='D'),
        'y': [100 + i*0.5 for i in range(180)]  # sample increasing data
    }
    df = pd.DataFrame(data)

    forecast = forecast_stock_price(df)
    plot_forecast(df, forecast)
    report = generate_report(forecast)
    print(report)
