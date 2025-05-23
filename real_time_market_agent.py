import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from dotenv import load_dotenv

import yfinance as yf
import requests
from newsapi import NewsApiClient
from tavily import TavilyClient
from transformers import pipeline


# ---------------------------- ENVIRONMENT ----------------------------
load_dotenv()

NEWSAPI_KEY       = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY    = os.getenv("GOOGLE_API_KEY")

if not all([
    NEWSAPI_KEY,
      ALPHA_VANTAGE_KEY, TAVILY_API_KEY, GOOGLE_API_KEY]):
    raise RuntimeError("Missing one or more required API keys in .env")

# ---------------------------- AGENT CLASS ----------------------------
class RealTimeMarketAgent:
    def __init__(self):
        self.newsapi   = NewsApiClient(api_key=NEWSAPI_KEY)
        self.av_key    = ALPHA_VANTAGE_KEY
        self.tavily    = TavilyClient(api_key=TAVILY_API_KEY)
        self.sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    def get_yahoo_finance_news(self, count: int = 5) -> List[Dict]:
        return self._search_news("site:finance.yahoo.com/news", "Yahoo Finance", count)

    def get_cnbc_news(self, count: int = 5) -> List[Dict]:
        return self._search_news("site:cnbc.com/finance", "CNBC", count)

    def get_reuters_news(self, count: int = 5) -> List[Dict]:
        return self._search_news("site:reuters.com/business", "Reuters", count)

    def _search_news(self, query: str, source: str, count: int = 5) -> List[Dict]:
        resp = self.tavily.search(query)
        items = list(resp.values()) if isinstance(resp, dict) else resp
        return [self._normalize_item(r, source) for r in items[:count]]

    def _normalize_item(self, r, source: str) -> Dict:
        if isinstance(r, dict):
            return {
                "title": r.get("title", ""),
                "source": source,
                "url": r.get("url", ""),
                "publishedAt": r.get("publishedAt", "")[:10]
            }
        else:
            return {
                "title": r,
                "source": source,
                "url": r,
                "publishedAt": ""
            }

    def get_realtime_price(self, ticker: str) -> Dict:
        tk = yf.Ticker(ticker)
        df = tk.history(period="2d", interval="1d")
        today, prior = df.iloc[-1], df.iloc[-2]
        pct = (today["Close"] - prior["Close"]) / prior["Close"] * 100
        return {
            "price": round(today["Close"], 2),
            "change_pct": round(pct, 2),
            "currency": tk.info.get("currency", "USD")
        }

    def get_intraday_events(self, ticker: str) -> Dict:
        url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
            f"&symbol={ticker}&interval=60min&apikey={self.av_key}"
        )
        ts = requests.get(url).json().get("Time Series (60min)", {})
        times = sorted(ts.keys())[-2:]
        latest, prev = ts[times[-1]], ts[times[-2]]
        return {
            "latest_time": times[-1],
            "latest_volume": int(latest["5. volume"]),
            "prev_volume": int(prev["5. volume"])
        }

    def get_latest_news(self, ticker: str, days: int = 1) -> List[Dict]:
        name = yf.Ticker(ticker).info.get("shortName", ticker)
        since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        res = self.newsapi.get_everything(
            q=f'"{ticker}" OR "{name}"',
            from_param=since,
            sort_by="relevancy",
            language="en",
            page_size=5
        )
        return [{
            "title": a["title"],
            "source": a["source"]["name"],
            "url": a["url"],
            "publishedAt": a["publishedAt"][:10],
            "description": a.get("description", "")
        } for a in res.get("articles", [])]

    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        results = []
        for txt in texts:
            if not txt:
                results.append({"label": "NEUTRAL", "score": 0.0})
            else:
                out = self.sentiment(txt[:512])[0]
                results.append({"label": out["label"], "score": out["score"]})
        return results

    def summarize(self, ticker: str) -> str:
        price = self.get_realtime_price(ticker)
        volume = self.get_intraday_events(ticker)
        api_news = self.get_latest_news(ticker)
        all_news = api_news + self.get_yahoo_finance_news() + self.get_cnbc_news() + self.get_reuters_news()
        news_items = [n for n in all_news if n.get("title")][:5]
        sentiments = self.analyze_sentiment([n["title"] for n in news_items])

        lines = [
            f"{ticker.upper()}: {price['price']} {price['currency']} ({price['change_pct']}% today)",
            f"Volume last hour: {volume['latest_volume']} (\u0394 {volume['latest_volume'] - volume['prev_volume']})",
            "Top 5 news headlines and sentiment:"
        ]
        for n, s in zip(news_items, sentiments):
            emoji = "\U0001F53A" if "POS" in s["label"].upper() or s["label"].startswith("5") else "\U0001F53B"
            lines.append(f"{emoji} {n['title']} ({n['source']}, {n['publishedAt']})")

        return "\n".join(lines)

# ---------------------------- LANGCHAIN AGENT WRAPPER ----------------------------

from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# Build tools from existing RealTimeMarketAgent
agent_instance = RealTimeMarketAgent()

from langchain_core.tools import StructuredTool


def get_buy_signal(ticker: str) -> str:
    price_info = agent_instance.get_realtime_price(ticker)
    change_pct = price_info["change_pct"]
    news = agent_instance.get_latest_news(ticker)
    sentiments = agent_instance.analyze_sentiment([n["title"] for n in news])
    pos_count = sum(1 for s in sentiments if "POS" in s["label"].upper() or s["label"].startswith("5"))
    neg_count = sum(1 for s in sentiments if "NEG" in s["label"].upper() or s["label"].startswith("1"))

    if change_pct > 0 and pos_count > neg_count:
        return f"🔺 BUY signal for {ticker}: Positive trend ({change_pct:+.2f}%), sentiment is mostly positive."
    elif change_pct < 0 and neg_count > pos_count:
        return f"🔻 SELL signal for {ticker}: Negative trend ({change_pct:+.2f}%), sentiment is mostly negative."
    else:
        return f"⏸ HOLD signal for {ticker}: No clear trend or mixed sentiment."


tools = [
    Tool(name="get_realtime_price", func=lambda x: str(agent_instance.get_realtime_price(x)), description="Get current stock price and daily change."),
    Tool(name="get_intraday_events", func=lambda x: str(agent_instance.get_intraday_events(x)), description="Get intraday trading volume changes."),
    Tool(name="get_latest_news", func=lambda x: str(agent_instance.get_latest_news(x)), description="Get latest company news."),
    Tool(name="summarize_market", func=agent_instance.summarize, description="Generate a full market summary for a stock ticker."),
    Tool(name="get_buy_signal", func=get_buy_signal, description="Returns BUY/SELL/HOLD signal based on price change and sentiment.")
]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a financial assistant. Use the tools provided to give accurate and up-to-date market answers."),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    
    ("human", "{input}")
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# ---------------------------- MAIN ----------------------------

if __name__ == "__main__":
    question = input("Enter your market question (e.g. 'Summarize TSLA'): ")
    result = agent_executor.invoke({"input": question})
    print("Market Assistant Result:" + "="*40)
    if isinstance(result, dict) and "output" in result:
        print(result["output"])
    else:
        print(result)

