import os
import re
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType

# === Load environment ===
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# === Utility functions ===

def parse_yearly_data(text):
    pattern = r"(\d{4})[^0-9]{1,10}?([\d.,]+)[\s]?[BbMm]?"
    matches = re.findall(pattern, text)
    data = {}
    for year, value in matches:
        value = float(value.replace(',', ''))
        data[int(year)] = value
    return dict(sorted(data.items()))

def forecast_next_value(data: dict):
    years = list(data.keys())
    values = list(data.values())
    if len(years) < 2:
        raise ValueError("Need at least 2 years of data to forecast")

    delta = values[-1] - values[-2]
    next_year = years[-1] + 1
    next_value = values[-1] + delta
    data[next_year] = next_value
    return data

def plot_forecast(data: dict, title="Financial Forecast"):
    years = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    plt.plot(years, values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("forecast.png")
    plt.close()


# === Tool function ===

def forecast_tool_func(query: str) -> str:
    data = parse_yearly_data(query)
    forecasted = forecast_next_value(data)
    plot_forecast(forecasted, title="Forecast based on financial data")
    return f"📈 Forecast complete. Predicted value for {max(forecasted.keys())}: {forecasted[max(forecasted.keys())]:,.2f}. Plot saved as forecast.png."


# === LangChain Tool ===

forecast_tool = Tool(
    name="FinancialForecaster",
    func=forecast_tool_func,
    description="Use this tool to forecast financial metrics based on yearly values given in the user's query. It parses numerical data and extrapolates one year forward, also saving a plot."
)


# === LangChain Agent ===

dc_agent = initialize_agent(
    tools=[forecast_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# === Entry point ===

if __name__ == "__main__":
    query = "Apple's revenue was 260B in 2020, 320B in 2021, 350B in 2022, and 400B in 2023"
    result = dc_agent.run(query)
    print("\n Agent Response:\n", result)

