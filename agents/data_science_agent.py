import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import yfinance as yf
import re
import tabula
import PyPDF2
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataScienceAgent')

class DataScienceAgent:
    """
    A comprehensive data science agent that can perform market analysis,
    trend analysis, and predictive modeling.
    """
    
    def __init__(self):
        """Initialize the DataScienceAgent with default parameters"""
        logger.info("Initializing DataScienceAgent")
        self.data = None
        self.ticker = None
        self.forecast_data = None
        self.model = None
        
    # Task 1: Data Extraction Methods
    def load_stock_data(self, ticker, period="1y", interval="1d"):
        """
        Load stock data for a specified ticker and period
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'MSFT')
            period (str): Time period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', etc.)
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            pd.DataFrame: DataFrame containing the stock data
        """
        logger.info(f"Loading stock data for {ticker} over {period} with {interval} interval")
        self.ticker = ticker
        try:
            stock = yf.Ticker(ticker)
            self.data = stock.history(period=period, interval=interval)
            if self.data.empty:
                logger.error(f"No data found for ticker {ticker}")
                return None
            
            # Clean and prepare data
            self.data = self.data.reset_index()
            if 'Date' not in self.data.columns and 'Datetime' in self.data.columns:
                self.data = self.data.rename(columns={'Datetime': 'Date'})
                
            logger.info(f"Successfully loaded {len(self.data)} records for {ticker}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            return None
    
    def extract_from_pdf(self, pdf_file, pages='all'):
        """
        Extract tables from PDF financial reports
        
        Args:
            pdf_file (str or file object): Path or file object of the PDF
            pages (str or list): Pages to extract tables from ('all' or list of page numbers)
            
        Returns:
            list: List of extracted pandas DataFrames
        """
        logger.info(f"Extracting tables from PDF (pages: {pages})")
        try:
            tables = tabula.read_pdf(pdf_file, pages=pages, multiple_tables=True)
            logger.info(f"Successfully extracted {len(tables)} tables from PDF")
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            return []
    
    def extract_text_from_pdf(self, pdf_file):
        """
        Extract full text from a PDF file
        
        Args:
            pdf_file (str or file object): Path or file object of the PDF
            
        Returns:
            str: Extracted text from PDF
        """
        logger.info("Extracting text from PDF")
        text = ""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            logger.info(f"Successfully extracted text from {len(reader.pages)} pages")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    # Task 2: Data Analysis Methods
    def calculate_basic_stats(self):
        """
        Calculate basic statistics on stock data
        
        Returns:
            dict: Dictionary containing statistical measures
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for analysis")
            return None
            
        logger.info("Calculating basic statistics")
        
        try:
            # Calculate daily returns
            self.data['Daily_Return'] = self.data['Close'].pct_change() * 100
            
            # Calculate various metrics
            stats = {
                'ticker': self.ticker,
                'start_date': self.data['Date'].iloc[0],
                'end_date': self.data['Date'].iloc[-1],
                'days': len(self.data),
                'start_price': self.data['Close'].iloc[0],
                'end_price': self.data['Close'].iloc[-1],
                'min_price': self.data['Close'].min(),
                'max_price': self.data['Close'].max(),
                'price_change': self.data['Close'].iloc[-1] - self.data['Close'].iloc[0],
                'price_change_pct': ((self.data['Close'].iloc[-1] / self.data['Close'].iloc[0]) - 1) * 100,
                'avg_daily_return': self.data['Daily_Return'].mean(),
                'volatility': self.data['Daily_Return'].std(),
                'max_daily_gain': self.data['Daily_Return'].max(),
                'max_daily_loss': self.data['Daily_Return'].min(),
                'volume_avg': self.data['Volume'].mean(),
                'volume_max': self.data['Volume'].max()
            }
            
            # Calculate moving averages
            self.data['MA50'] = self.data['Close'].rolling(window=50).mean()
            self.data['MA200'] = self.data['Close'].rolling(window=200).mean()
            
            logger.info("Successfully calculated basic statistics")
            return stats
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return None
    
    def analyze_trends(self):
        """
        Analyze trends in the stock data
        
        Returns:
            dict: Dictionary containing trend analysis
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for trend analysis")
            return None
            
        logger.info("Analyzing trends")
        
        try:
            # Calculate technical indicators
            # RSI (Relative Strength Index)
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (Moving Average Convergence Divergence)
            self.data['EMA12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
            self.data['EMA26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
            self.data['MACD'] = self.data['EMA12'] - self.data['EMA26']
            self.data['Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
            self.data['20dSTD'] = self.data['Close'].rolling(window=20).std()
            self.data['Upper_Band'] = self.data['MA20'] + (self.data['20dSTD'] * 2)
            self.data['Lower_Band'] = self.data['MA20'] - (self.data['20dSTD'] * 2)
            
            # Trend identification
            current_price = self.data['Close'].iloc[-1]
            ma50 = self.data['MA50'].iloc[-1]
            ma200 = self.data['MA200'].iloc[-1]
            
            # Identify current trend
            if current_price > ma50 > ma200:
                trend = "Strong Uptrend"
            elif current_price > ma50 and ma50 < ma200:
                trend = "Potential Reversal Upward"
            elif current_price < ma50 < ma200:
                trend = "Strong Downtrend"
            elif current_price < ma50 and ma50 > ma200:
                trend = "Potential Reversal Downward"
            else:
                trend = "Mixed/Sideways"
                
            # RSI interpretation
            rsi = self.data['RSI'].iloc[-1]
            if np.isnan(rsi):
                rsi_status = "Unknown"
            elif rsi > 70:
                rsi_status = "Overbought"
            elif rsi < 30:
                rsi_status = "Oversold"
            else:
                rsi_status = "Neutral"
                
            # Volatility assessment
            recent_volatility = self.data['Daily_Return'].tail(30).std()
            overall_volatility = self.data['Daily_Return'].std()
            
            if recent_volatility > overall_volatility * 1.5:
                volatility_status = "Increased Volatility"
            elif recent_volatility < overall_volatility * 0.5:
                volatility_status = "Decreased Volatility"
            else:
                volatility_status = "Normal Volatility"
                
            # Compile trend analysis results
            trend_analysis = {
                'current_trend': trend,
                'rsi_value': rsi,
                'rsi_status': rsi_status,
                'macd': self.data['MACD'].iloc[-1],
                'signal': self.data['Signal'].iloc[-1],
                'volatility_status': volatility_status,
                'price_to_ma50_ratio': current_price / ma50 if not np.isnan(ma50) else None,
                'price_to_ma200_ratio': current_price / ma200 if not np.isnan(ma200) else None
            }
            
            logger.info("Successfully completed trend analysis")
            return trend_analysis
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return None
    
    # Task 3: Predictive Modeling Methods
    def forecast_with_arima(self, steps=90):
        """
        Create a stock price forecast using ARIMA model
        
        Args:
            steps (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: DataFrame containing forecast data
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for ARIMA forecasting")
            return None
            
        logger.info(f"Forecasting with ARIMA for {steps} steps")
        
        try:
            # Prepare data for ARIMA
            y = self.data['Close'].values
            
            # Fit ARIMA model - using a simple model (1,1,1) for demonstration
            # In production, parameters should be properly selected using AIC/BIC
            model = ARIMA(y, order=(1, 1, 1))
            self.model = model.fit()
            
            # Forecast
            forecast_values = self.model.forecast(steps=steps)
            
            # Create date range for forecast
            last_date = self.data['Date'].iloc[-1]
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, '%Y-%m-%d')
                
            date_range = pd.date_range(start=last_date + timedelta(days=1), periods=steps)
            
            # Create forecast DataFrame
            self.forecast_data = pd.DataFrame({
                'Date': date_range,
                'Forecast': forecast_values
            })
            
            # Add confidence intervals based on model's capabilities
            if hasattr(self.model, 'get_forecast'):
                forecast_obj = self.model.get_forecast(steps=steps)
                forecast_ci = forecast_obj.conf_int()
                self.forecast_data['Lower_CI'] = forecast_ci.iloc[:, 0].values
                self.forecast_data['Upper_CI'] = forecast_ci.iloc[:, 1].values
            else:
                # Simple confidence interval based on historical volatility
                std_dev = self.data['Close'].pct_change().std() * self.forecast_data['Forecast']
                self.forecast_data['Lower_CI'] = self.forecast_data['Forecast'] - (1.96 * std_dev)
                self.forecast_data['Upper_CI'] = self.forecast_data['Forecast'] + (1.96 * std_dev)
            
            logger.info(f"Successfully created ARIMA forecast for {steps} days")
            return self.forecast_data
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {e}")
            return None
    
    def forecast_with_prophet(self, periods=90):
        """
        Create a stock price forecast using Facebook Prophet
        
        Args:
            periods (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: DataFrame containing forecast data
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for Prophet forecasting")
            return None
            
        logger.info(f"Forecasting with Prophet for {periods} periods")
        
        try:
            # Prepare data for Prophet
            prophet_data = self.data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            
            # Initialize and fit the model
            model = Prophet(daily_seasonality=True)
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Save forecast data
            self.forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower_CI', 'yhat_upper': 'Upper_CI'}
            )
            
            # Filter to only include the forecast period
            last_date = self.data['Date'].iloc[-1]
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, '%Y-%m-%d')
            
            self.model = model  # Save the model for potential future use
            
            logger.info(f"Successfully created Prophet forecast for {periods} days")
            return self.forecast_data
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {e}")
            return None
    
    def evaluate_forecast_accuracy(self, test_size=30):
        """
        Evaluate forecast accuracy using historical data
        
        Args:
            test_size (int): Number of days to use for testing
            
        Returns:
            dict: Dictionary containing accuracy metrics
        """
        if self.data is None or self.data.empty or len(self.data) <= test_size:
            logger.error("Insufficient data for forecast evaluation")
            return None
            
        logger.info(f"Evaluating forecast accuracy with test size {test_size}")
        
        try:
            # Split data into train and test
            train_data = self.data.iloc[:-test_size]
            test_data = self.data.iloc[-test_size:]
            
            # Store original data
            original_data = self.data
            
            # Set training data temporarily
            self.data = train_data
            
            # Generate forecast with ARIMA
            _ = self.forecast_with_arima(steps=test_size)
            arima_forecast = self.forecast_data['Forecast'].values
            
            # Generate forecast with Prophet
            _ = self.forecast_with_prophet(periods=test_size)
            prophet_forecast = self.forecast_data['Forecast'].values[-test_size:]
            
            # Actual values
            actual = test_data['Close'].values
            
            # Calculate metrics
            metrics = {
                'ARIMA': {
                    'MAE': mean_absolute_error(actual, arima_forecast),
                    'RMSE': np.sqrt(mean_squared_error(actual, arima_forecast)),
                    'MAPE': np.mean(np.abs((actual - arima_forecast) / actual)) * 100
                },
                'Prophet': {
                    'MAE': mean_absolute_error(actual, prophet_forecast),
                    'RMSE': np.sqrt(mean_squared_error(actual, prophet_forecast)),
                    'MAPE': np.mean(np.abs((actual - prophet_forecast) / actual)) * 100
                }
            }
            
            # Restore original data
            self.data = original_data
            
            logger.info("Successfully evaluated forecast accuracy")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating forecast accuracy: {e}")
            return None
    
    # Task 4: Visualization Methods
    def plot_stock_history(self, include_volume=True):
        """
        Create an interactive plot of stock price history
        
        Args:
            include_volume (bool): Whether to include volume in the plot
            
        Returns:
            plotly.Figure: Interactive Plotly figure
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for visualization")
            return None
            
        logger.info("Creating stock history plot")
        
        try:
            # Create subplots
            if include_volume:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.1, 
                                    row_heights=[0.7, 0.3])
            else:
                fig = make_subplots(rows=1, cols=1)
            
            # Add price trace
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=self.data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            if 'MA50' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['MA50'],
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='orange', dash='dash')
                    ),
                    row=1, col=1
                )
                
            if 'MA200' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['MA200'],
                        mode='lines',
                        name='200-Day MA',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=1
                )
                
            # Add Bollinger Bands if available
            if all(col in self.data.columns for col in ['Upper_Band', 'Lower_Band']):
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['Upper_Band'],
                        mode='lines',
                        name='Upper Band',
                        line=dict(color='lightgray', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['Lower_Band'],
                        mode='lines',
                        name='Lower Band',
                        line=dict(color='lightgray', width=1),
                        fill='tonexty',
                        fillcolor='rgba(200, 200, 200, 0.2)'
                    ),
                    row=1, col=1
                )
            
            # Add volume if requested
            if include_volume:
                fig.add_trace(
                    go.Bar(
                        x=self.data['Date'],
                        y=self.data['Volume'],
                        name='Volume',
                        marker=dict(color='rgba(0, 100, 80, 0.5)')
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{self.ticker} Stock Price History',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=800 if include_volume else 600,
                legend=dict(orientation='h', y=1.02),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            if include_volume:
                fig.update_yaxes(title_text='Volume', row=2, col=1)
                
            logger.info("Successfully created stock history visualization")
            return fig
        except Exception as e:
            logger.error(f"Error creating stock history plot: {e}")
            return None
    
    def plot_forecast(self):
        """
        Create an interactive plot of the stock price forecast
        
        Returns:
            plotly.Figure: Interactive Plotly figure with forecast
        """
        if self.data is None or self.data.empty or self.forecast_data is None:
            logger.error("No data available for forecast visualization")
            return None
            
        logger.info("Creating forecast visualization")
        
        try:
            # Create figure
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=self.data['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                )
            )
            
            # Add forecast
            fig.add_trace(
                go.Scatter(
                    x=self.forecast_data['Date'],
                    y=self.forecast_data['Forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                )
            )
            
            # Add confidence intervals
            if all(col in self.forecast_data.columns for col in ['Lower_CI', 'Upper_CI']):
                fig.add_trace(
                    go.Scatter(
                        x=self.forecast_data['Date'],
                        y=self.forecast_data['Upper_CI'],
                        mode='lines',
                        name='Upper CI',
                        line=dict(color='rgba(255, 0, 0, 0.2)', width=0)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=self.forecast_data['Date'],
                        y=self.forecast_data['Lower_CI'],
                        mode='lines',
                        name='Lower CI',
                        line=dict(color='rgba(255, 0, 0, 0.2)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.1)'
                    )
                )
            
            # Add vertical line at current date
            last_historical_date = self.data['Date'].iloc[-1]
            
            fig.add_vline(
                x=last_historical_date,
                line_width=2,
                line_dash="dash",
                line_color="green",
                annotation_text="Forecast Start",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                title=f'{self.ticker} Stock Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=600,
                legend=dict(orientation='h', y=1.02),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            logger.info("Successfully created forecast visualization")
            return fig
        except Exception as e:
            logger.error(f"Error creating forecast plot: {e}")
            return None
    
    def plot_technical_indicators(self):
        """
        Create an interactive plot with technical indicators
        
        Returns:
            plotly.Figure: Interactive Plotly figure with indicators
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for technical indicators visualization")
            return None
            
        logger.info("Creating technical indicators visualization")
        
        try:
            # Create subplots: price, RSI, MACD
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, 
                                row_heights=[0.6, 0.2, 0.2])
            
            # Plot price and MAs
            fig.add_trace(
                go.Scatter(
                    x=self.data['Date'],
                    y=self.data['Close'],
                    mode='lines',
                    name='Close',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add moving averages if available
            if 'MA50' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['MA50'],
                        mode='lines',
                        name='50-Day MA',
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )
                
            if 'MA200' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['MA200'],
                        mode='lines',
                        name='200-Day MA',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
            
            # Add RSI if available
            if 'RSI' in self.data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                
                # Add RSI threshold lines
                fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
            
            # Add MACD if available
            if all(col in self.data.columns for col in ['MACD', 'Signal']):
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=self.data['Date'],
                        y=self.data['Signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red')
                    ),
                    row=3, col=1
                )
                
                # Calculate MACD histogram
                self.data['MACD_Hist'] = self.data['MACD'] - self.data['Signal']
                
                # Add MACD histogram
                colors = ['green' if val >= 0 else 'red' for val in self.data['MACD_Hist']]
                
                fig.add_trace(
                    go.Bar(
                        x=self.data['Date'],
                        y=self.data['MACD_Hist'],
                        name='MACD Histogram',
                        marker_color=colors
                    ),
                    row=3, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{self.ticker} Technical Indicators',
                height=900,
                legend=dict(orientation='h', y=1.02),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text='Price ($)', row=1, col=1)
            fig.update_yaxes(title_text='RSI', row=2, col=1)
            fig.update_yaxes(title_text='MACD', row=3, col=1)
            
            logger.info("Successfully created technical indicators visualization")
            return fig
        except Exception as e:
            logger.error(f"Error creating technical indicators plot: {e}")
            return None
    
    # Task 5: Report Generation Methods
    def generate_market_analysis_report(self):
        """
        Generate a comprehensive market analysis report
        
        Returns:
            str: Markdown-formatted report
        """
        if self.data is None or self.data.empty:
            logger.error("No data available for report generation")
            return "No data available for analysis."
            
        logger.info("Generating market analysis report")
        
        try:
            # Fetch basic stats and trend analysis
            stats = self.calculate_basic_stats()
            trends = self.analyze_trends()
            
            if stats is None or trends is None:
                return "Error generating analysis data."
            
            # Format price change with + or - sign
            price_change = stats['price_change']
            price_change_pct = stats['price_change_pct']
            price_change_str = f"+${price_change:.2f}" if price_change >= 0 else f"-${abs(price_change):.2f}"
            price_change_pct_str = f"+{price_change_pct:.2f}%" if price_change_pct >= 0 else f"-{abs(price_change_pct):.2f}%"
            
            # Generate report
            report = f"""# Market Analysis Report: {self.ticker}

## Executive Summary
This report provides a comprehensive analysis of {self.ticker} stock performance from {stats['start_date'].strftime('%B %d, %Y') if isinstance(stats['start_date'], datetime) else stats['start_date']} to {stats['end_date'].strftime('%B %d, %Y') if isinstance(stats['end_date'], datetime) else stats['end_date']}.

**Current Price:** ${stats['end_price']:.2f}  
**Price Change:** {price_change_str} ({price_change_pct_str})  
**Trading Volume:** {stats['avg_volume']:,.0f} shares (daily average)
**Volatility:** {stats['volatility']:.2f}%

## Price Analysis
- **Opening Price:** ${stats['start_price']:.2f}
- **Closing Price:** ${stats['end_price']:.2f}
- **Highest Price:** ${stats['max_price']:.2f} (reached on {stats['max_price_date'].strftime('%B %d, %Y') if isinstance(stats['max_price_date'], datetime) else stats['max_price_date']})
- **Lowest Price:** ${stats['min_price']:.2f} (reached on {stats['min_price_date'].strftime('%B %d, %Y') if isinstance(stats['min_price_date'], datetime) else stats['min_price_date']})

## Market Trend Analysis
{trends['summary']}

### Key Trend Indicators:
- **Moving Average (50-day):** ${trends['ma_50']:.2f}
- **Moving Average (200-day):** ${trends['ma_200']:.2f}
- **RSI (14-day):** {trends['rsi']:.2f}
- **MACD:** {trends['macd']:.2f}

## Volume Analysis
The average daily trading volume was {stats['avg_volume']:,.0f} shares. {trends['volume_analysis']}

## Technical Indicators
{trends['technical_indicators_summary']}

## Risk Assessment
Based on historical volatility of {stats['volatility']:.2f}% and current market conditions, {self.ticker} presents a {trends['risk_assessment']} risk investment opportunity.

## Recommendation
{trends['recommendation']}

## Disclaimer
This report is generated for informational purposes only and does not constitute investment advice. All investment decisions should be made after consulting with a qualified financial advisor and conducting thorough research.

Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
"""
        
            logger.info(f"Market analysis report for {self.ticker} generated successfully")
            return report
        
        except Exception as e:
            logger.error(f"Error generating market analysis report: {str(e)}")
            return f"Error generating report: {str(e)}"