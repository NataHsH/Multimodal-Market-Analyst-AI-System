import pandas as pd
from datetime import datetime, timedelta
from forecast import plot_interactive, generate_forecast
from report_generator import generate_report, generate_trend_analysis

END_DATE = datetime(2024, 12, 31)  # Upper limit for available data


def parse_request(request: str):
    """
    Parse the user request to extract company name, date range, and action type.
    Supports keywords:
    - 'letztes Jahr' (last year)
    - 'nächstes Quartal' (next quarter)
    - 'trend analyse' (trend analysis for the last year)
    - 'prognose' (forecast for the next quarter)
    """
    request_lower = request.lower()
    
    # Default company and date range
    company = "Microsoft"  # Default company
    start_date = datetime(2020, 1, 1).date()
    end_date = END_DATE.date()
    action = ""

    # List of target companies
    companies = ['microsoft', 'apple', 'google', 'nvidia', 'meta']
    for c in companies:
        if c in request_lower:
            company = c.capitalize()
            break

    # Handle 'letztes Jahr' and 'trend analyse'
    if "letztes jahr" in request_lower or "trend analyse" in request_lower:
        last_year = datetime.now().year - 1
        start_date = datetime(last_year, 1, 1).date()
        end_date = datetime(last_year, 12, 31).date()
        action = "trend"

    # Handle 'nächstes Quartal' and 'prognose'
    if "nächstes quartal" in request_lower or "prognose" in request_lower:
        today = datetime.now()
        current_quarter = (today.month - 1) // 3 + 1
        next_quarter = current_quarter + 1
        next_quarter_year = today.year
        if next_quarter > 4:
            next_quarter = 1
            next_quarter_year += 1
        quarter_months = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}
        start_month, end_month = quarter_months[next_quarter]
        start_date = datetime(next_quarter_year, start_month, 1).date()
        end_day = 31 if end_month == 12 else (datetime(next_quarter_year, end_month + 1, 1) - timedelta(days=1)).day
        end_date = datetime(next_quarter_year, end_month, end_day).date()
        action = "forecast"

    # Limit end date to the predefined END_DATE
    if end_date > END_DATE.date():
        end_date = END_DATE.date()
    if start_date > end_date:
        start_date = end_date

    return company, start_date, end_date, action


def handle_request(df: pd.DataFrame, request: str):
    """
    Process user request: parse, filter data, generate report, and plot.
    Returns structured output with company, period, action, and report text.
    """
    company, start_date, end_date, action = parse_request(request)

    # Filter data by company
    if 'company' in df.columns:
        df_filtered = df[df['company'].str.lower() == company.lower()]
    else:
        df_filtered = df.copy()

    # Filter data by date range
    if 'date' in df_filtered.columns:
        mask = (df_filtered['date'] >= pd.to_datetime(start_date)) & (df_filtered['date'] <= pd.to_datetime(end_date))
        df_filtered = df_filtered.loc[mask]

    # If no data is available after filtering, return a default response
    if df_filtered.empty:
        return {
            "company": company,
            "period": f"{start_date} to {end_date}",
            "action": action,
            "report": "No data available for the requested period and company."
        }

    # Determine action and call respective function
    if action == "trend":
        report = generate_trend_analysis(df_filtered)
    elif action == "forecast":
        report = generate_forecast(df_filtered)
    else:
        report = generate_report(df_filtered)

    # Generate interactive plot
    plot_interactive(df_filtered)

    return {
        "company": company,
        "period": f"{start_date} to {end_date}",
        "action": action,
        "report": report
    }
