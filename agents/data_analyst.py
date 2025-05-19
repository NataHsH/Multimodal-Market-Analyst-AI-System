from datetime import datetime
import pandas as pd
from data_extraction_chromaDB import extract_data_from_chromadb
from forecast import forecast_stock_price, plot_forecast
from forecast_arima import forecast_arima, plot_forecast_arima
from report_generator import generate_report

def get_last_n_years_period(n=5):
    """
    Return start and end year strings for the last n full years.
    """
    current_year = datetime.now().year
    start_year = current_year - n
    return str(start_year), str(current_year - 1)  # например, ('2020', '2024')

def main(companies=None, years=5):
    if companies is None:
        companies = ["Apple", "Microsoft", "Google", "NVIDIA", "Meta"]
    
    start_year, end_year = get_last_n_years_period(years)
    
    all_data = []
    for company in companies:
        for year in range(int(start_year), int(end_year) + 1):
            df = extract_data_from_chromadb(company_name=company, data_type="stock_price", time_frame=str(year))
            if not df.empty:
                all_data.append(df)
    
    if not all_data:
        raise ValueError("No data extracted for the selected companies and period.")
    
    df_all = pd.concat(all_data).reset_index(drop=True)

    # Подготовка для Prophet
    df_all.rename(columns={'date': 'ds', 'stock_price': 'y'}, inplace=True)
    
    # Пример: прогноз и визуализация для Microsoft (параметр можно сделать)
    company_to_forecast = "Microsoft"
    df_company = df_all[df_all['company'].str.lower() == company_to_forecast.lower()]
    
    if df_company.empty:
        raise ValueError(f"No {company_to_forecast} data available for forecasting.")
    
    # Прогноз Prophet
    forecast = forecast_stock_price(df_company)
    plot_forecast(df_company, forecast)
    
    # Прогноз ARIMA
    forecast_arima_df = forecast_arima(df_company)
    plot_forecast_arima(df_company, forecast_arima_df)
    
    # Генерация отчёта по прогнозу Prophet
    forecast_report_df = forecast.rename(columns={
        'ds': 'date',
        'yhat': 'stock_price',
        'yhat_lower': 'lower_bound',
        'yhat_upper': 'upper_bound'
    })[['date', 'stock_price', 'lower_bound', 'upper_bound']]
    
    generate_report(forecast_report_df)

if __name__ == "__main__":
    main()
