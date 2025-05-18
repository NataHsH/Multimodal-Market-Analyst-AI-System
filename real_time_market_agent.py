# real_time_market_agent.py

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import yfinance as yf
import requests
from newsapi import NewsApiClient
from tavily import TavilyClient
from transformers import pipeline

# ---------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------
load_dotenv()
NEWSAPI_KEY       = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
if not NEWSAPI_KEY or not ALPHA_VANTAGE_KEY or not TAVILY_API_KEY:
    raise RuntimeError("One or more required API keys are missing in .env")

class RealTimeMarketAgent:
    """
    Agent to fetch real-time prices, volume, API news, web-search news via Tavily,
    and perform sentiment analysis.
    """

    def __init__(self):
        # Initialize API clients and NLP pipeline
        self.newsapi   = NewsApiClient(api_key=NEWSAPI_KEY)
        self.av_key    = ALPHA_VANTAGE_KEY
        self.tavily    = TavilyClient(api_key=TAVILY_API_KEY)
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

    # -----------------------------------------------------------------
    # Section: Web-Search News via Tavily
    # -----------------------------------------------------------------
    def search_tavily(self, query: str, count: int = 5) -> List:
        """
        Perform a Tavily search and return up to `count` raw result items.
        Handles both list and dict response formats.
        """
        resp = self.tavily.search(query)
        items = list(resp.values()) if isinstance(resp, dict) else resp
        return items[:count]

    def _normalize_item(self, r, source: str) -> Dict:
        """
        Normalize a single Tavily result into a uniform dict:
        - if r is dict: extract title/url/publishedAt
        - if r is str: treat it as URL (title=url, date empty)
        """
        if isinstance(r, dict):
            return {
                "title":       r.get("title", ""),
                "source":      source,
                "url":         r.get("url", ""),
                "publishedAt": r.get("publishedAt", "")[:10]
            }
        else:
            return {
                "title":       r,
                "source":      source,
                "url":         r,
                "publishedAt": ""
            }

    def get_yahoo_finance_news(self, count: int = 5) -> List[Dict]:
        """Get latest Yahoo Finance headlines via Tavily."""
        results = self.search_tavily("site:finance.yahoo.com/news", count)
        return [self._normalize_item(r, "Yahoo Finance") for r in results]

    def get_cnbc_news(self, count: int = 5) -> List[Dict]:
        """Get latest CNBC headlines via Tavily."""
        results = self.search_tavily("site:cnbc.com/finance", count)
        return [self._normalize_item(r, "CNBC") for r in results]

    def get_reuters_news(self, count: int = 5) -> List[Dict]:
        """Get latest Reuters Business headlines via Tavily."""
        results = self.search_tavily("site:reuters.com/business", count)
        return [self._normalize_item(r, "Reuters") for r in results]

    # -----------------------------------------------------------------
    # Section: Price & Volume Data
    # -----------------------------------------------------------------
    def get_realtime_price(self, ticker: str) -> Dict:
        """
        Fetch closing prices for the last two days via yfinance,
        compute percentage change, and return price info.
        """
        tk    = yf.Ticker(ticker)
        df    = tk.history(period="2d", interval="1d")
        today, prior = df.iloc[-1], df.iloc[-2]
        pct   = (today["Close"] - prior["Close"]) / prior["Close"] * 100
        return {
            "price":      round(today["Close"], 2),
            "change_pct": round(pct, 2),
            "currency":   tk.info.get("currency", "USD")
        }

    def get_intraday_events(self, ticker: str) -> Dict:
        """
        Query Alpha Vantage for intraday (60min) series,
        extract the last two volumes for comparison.
        """
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_INTRADAY&symbol={ticker}"
            f"&interval=60min&apikey={self.av_key}"
        )
        js    = requests.get(url).json()
        ts    = js.get("Time Series (60min)", {})
        times = sorted(ts.keys())[-2:]
        latest, prev = ts[times[-1]], ts[times[-2]]
        return {
            "latest_time":  times[-1],
            "latest_volume": int(latest["5. volume"]),
            "prev_volume":   int(prev["5. volume"])
        }

    # -----------------------------------------------------------------
    # Section: API-Based News (NewsAPI)
    # -----------------------------------------------------------------
    def get_latest_news(self, ticker: str, days: int = 1) -> List[Dict]:
        """Fetch up to 5 relevant news articles from NewsAPI."""
        name  = yf.Ticker(ticker).info.get("shortName", ticker)
        since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        res   = self.newsapi.get_everything(
            q=f'"{ticker}" OR "{name}"',
            from_param=since,
            sort_by="relevancy",
            language="en",
            page_size=5
        )
        return [
            {
                "title":       a["title"],
                "source":      a["source"]["name"],
                "url":         a["url"],
                "publishedAt": a["publishedAt"][:10],
                "description": a.get("description", "")
            }
            for a in res.get("articles", [])
        ]

    # -----------------------------------------------------------------
    # Section: Sentiment Analysis
    # -----------------------------------------------------------------
    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        """
        Compute sentiment labels and scores for a list of text snippets.
        Empty strings yield a default neutral sentiment.
        """
        results = []
        for txt in texts:
            if not txt:
                results.append({"label": "NEUTRAL", "score": 0.0})
            else:
                out = self.sentiment(txt[:512])[0]
                results.append({"label": out["label"], "score": out["score"]})
        return results

    # -----------------------------------------------------------------
    # Section: Summarize
    # -----------------------------------------------------------------
    def summarize(self, ticker: str) -> str:
        """
        Combine price, volume, multiple news sources, and sentiment
        into a single human-readable summary.
        """
        price    = self.get_realtime_price(ticker)
        volume   = self.get_intraday_events(ticker)
        api_news = self.get_latest_news(ticker)
        yf_news  = self.get_yahoo_finance_news()
        cnbc     = self.get_cnbc_news()
        reuters  = self.get_reuters_news()
        all_news = api_news + yf_news + cnbc + reuters

        art_news = [n for n in all_news if n.get("title")][:5]
        titles   = [n["title"] for n in art_news]
        sentiments = self.analyze_sentiment(titles)

        lines = [
            f"{ticker}: {price['price']} {price['currency']} ({price['change_pct']}% today).",
            f"Volume last hour: {volume['latest_volume']} (Δ {volume['latest_volume']-volume['prev_volume']}).",
            "Top 5 news & sentiment:"
        ]

        for art, sent in zip(art_news, sentiments):
            emoji = "🔺" if "POS" in sent["label"].upper() or sent["label"].startswith("5") else "🔻"
            lines.append(f"{emoji} {art['title']} ({art['source']}, {art['publishedAt']})")

        return "\n".join(lines)


if __name__ == "__main__":
    agent = RealTimeMarketAgent()

    # Smoke tests for news scrapers
    print("Yahoo Finance headlines:")
    for y in agent.get_yahoo_finance_news():
        print("•", y)

    print("\nCNBC headlines:")
    for c in agent.get_cnbc_news():
        print("•", c)

    print("\nReuters headlines:")
    for r in agent.get_reuters_news():
        print("•", r)

    # Smoke tests for Alpha Vantage and yfinance
    print("\nIntraday events for GOOGL:")
    print(agent.get_intraday_events("GOOGL"))

    print("\nRealtime price for GOOGL:")
    print(agent.get_realtime_price("GOOGL"))

    # Full summary
    print("\nSummary for GOOGL:")
    print(agent.summarize("GOOGL"))
