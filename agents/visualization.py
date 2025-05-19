# Data Visualization Module

import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Static Plot using Matplotlib
def plot_static(df):
    """
    Create a static plot of stock prices with confidence intervals using Matplotlib.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['stock_price'], label='Stock Price', color='blue')
    plt.fill_between(df['date'], df['lower_bound'], df['upper_bound'], color='blue', alpha=0.2, label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Over Time with Confidence Interval')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

# Interactive Plot using Plotly
def plot_interactive(df):
    """
    Create an interactive plot of stock prices with confidence intervals using Plotly.
    """
    fig = go.Figure()

    # Plot stock price line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['stock_price'],
        mode='lines',
        name='Stock Price',
        line=dict(color='blue')
    ))

    # Upper bound (invisible trace for filling)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['upper_bound'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Lower bound with fill to the previous trace (upper bound)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['lower_bound'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.2)',  # translucent blue
        line=dict(width=0),
        name='Confidence Interval',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title='Interactive Stock Price Chart with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        template='plotly_white'
    )

    fig.show()
