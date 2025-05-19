import pandas as pd

# Generate Analytical Report
def generate_report(df):
    """
    Generate a detailed textual report based on forecasted stock price data.
    """
    latest_price = df['stock_price'].iloc[-1]
    latest_date = df['date'].iloc[-1]
    lower_bound = df['lower_bound'].iloc[-1]
    upper_bound = df['upper_bound'].iloc[-1]

    trend = "increasing" if df['stock_price'].iloc[-1] > df['stock_price'].iloc[-2] else "decreasing"

    # Forecasted price as the mean of the last confidence interval
    forecasted_price = (lower_bound + upper_bound) / 2

    # Calculate price volatility as standard deviation over the observed period
    volatility = df['stock_price'].std()

    report = (
        f"As of {latest_date.date()}, the stock price is {latest_price:.2f}.\n"
        f"The forecast indicates a {trend} trend with a predicted price of {forecasted_price:.2f}.\n"
        f"The confidence interval ranges from {lower_bound:.2f} to {upper_bound:.2f}, "
        f"representing the range within which the actual price is expected to fall with high probability.\n"
        f"Observed price volatility during the period is {volatility:.2f}, indicating how much prices fluctuate.\n"
        f"Recommendation: Monitor market conditions closely, consider potential external factors "
        f"such as economic news or earnings reports that might impact the stock price. "
        f"Use the confidence interval to assess risk when making investment decisions."
    )
    print(report)
    return report
