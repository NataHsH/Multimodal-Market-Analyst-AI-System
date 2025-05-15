# real_time_market_agent.py

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import yfinance as yf
import requests
from newsapi import NewsApiClient
from transformers import pipeline

# Сначала подгружаем .env
load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

if NEWSAPI_KEY is None or ALPHA_VANTAGE_KEY is None:
    raise RuntimeError("Не получилось загрузить NEWSAPI_KEY или ALPHA_VANTAGE_KEY из .env")

class RealTimeMarketAgent:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
        self.av_key = ALPHA_VANTAGE_KEY
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )

    def get_realtime_price(self, ticker: str) -> Dict:
        tk = yf.Ticker(ticker)
        data = tk.history(period="2d", interval="1d")
        today, yesterday = data.iloc[-1], data.iloc[-2]
        change_pct = (today["Close"] - yesterday["Close"]) / yesterday["Close"] * 100
        return {
            "price": round(today["Close"], 2),
            "change_pct": round(change_pct, 2),
            "currency": tk.info.get("currency", "USD")
        }

    def get_intraday_events(self, ticker: str) -> Dict:
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_INTRADAY&symbol={ticker}"
            f"&interval=60min&apikey={self.av_key}"
        )
        js = requests.get(url).json()
        ts = js.get("Time Series (60min)", {})
        times = sorted(ts.keys())[-2:]
        latest, prev = ts[times[-1]], ts[times[-2]]
        return {
            "latest_time": times[-1],
            "latest_volume": int(latest["5. volume"]),
            "prev_volume": int(prev["5. volume"])
        }

    def get_latest_news(self, ticker: str, days: int = 1) -> List[Dict]:
        company = yf.Ticker(ticker).info.get("shortName", ticker)
        # формируем строку YYYY-MM-DD, как требует NewsAPI
        from_dt = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        res = self.newsapi.get_everything(
            q=f'"{ticker}" OR "{company}"',
            from_param=from_dt,
            sort_by="relevancy",
            language="en",
            page_size=5
        )
        articles = []
        for art in res.get("articles", []):
            articles.append({
                "title": art["title"],
                "source": art["source"]["name"],
                "url": art["url"],
                "publishedAt": art["publishedAt"][:10],
                "description": art.get("description", "")
            })
        return articles

    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        results = []
        for t in texts:
            out = self.sentiment(t[:512])[0]
            results.append({"label": out["label"], "score": out["score"]})
        return results

    def summarize(self, ticker: str) -> str:
        price_info = self.get_realtime_price(ticker)
        events = self.get_intraday_events(ticker)
        news = self.get_latest_news(ticker)

        sentiments = self.analyze_sentiment([n["title"] for n in news])

        lines = [
            f"{ticker}: {price_info['price']} {price_info['currency']} ({price_info['change_pct']}% сегодня).",
            f"Объём за последний час: {events['latest_volume']} (Δ {events['latest_volume'] - events['prev_volume']}).",
            "Ключевые новости:"
        ]
        for art, sent in zip(news, sentiments):
            emoji = "🔺" if "POS" in sent["label"].upper() or sent["label"].startswith("5") else "🔻"
            lines.append(f"{emoji} {art['title']} ({art['source']}, {art['publishedAt']})")

        return "\n".join(lines)

if __name__ == "__main__":
    # маленький тест
    print("NEWSAPI_KEY:", NEWSAPI_KEY)
    print("ALPHA_VANTAGE_KEY:", ALPHA_VANTAGE_KEY)
    agent = RealTimeMarketAgent()
    print(agent.summarize("GOOGL"))
