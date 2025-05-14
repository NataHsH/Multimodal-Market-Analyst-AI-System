import pdfplumber
import pandas as pd

def extract_tables_from_pdf(pdf_path):
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                all_tables.append(df)
    return all_tables

def clean_financial_table(df):
    df = df.dropna().copy()
    df.columns = [col.strip().lower() for col in df.columns]
    
    df = df.rename(columns={
        "quarter": "date", 
        "revenue": "value"
    })

    # Преобразуем дату и значение
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].str.replace(',', ''), errors='coerce')

    return df[["date", "value"]].dropna()

from prophet import Prophet

def forecast_with_prophet(df, periods=4):
    df_prophet = df.rename(columns={"date": "ds", "value": "y"})
    model = Prophet()
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=periods, freq='Q')
    forecast = model.predict(future)
    
    return model, forecast

import matplotlib.pyplot as plt

def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.title("Revenue Forecast")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()

def data_science_pipeline(pdf_path):
    tables = extract_tables_from_pdf(pdf_path)
    tables = extract_tables_from_pdf("10-Q4-2024-As-Filed.pdf")

    for i, table in enumerate(tables):
        print(f"\n--- Таблица {i+1} ---")
        print(table.head())
    
    for table in tables:
        df = clean_financial_table(table)
        if not df.empty and len(df) >= 6:  # простая фильтрация
            model, forecast = forecast_with_prophet(df)
            plot_forecast(model, forecast)
            #insight = generate_summary_insight(df, forecast)
            #print(insight)
            break

data_science_pipeline("10-Q4-2024-As-Filed.pdf")


