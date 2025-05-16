# tests/test_market_agent.py

import pytest
from real_time_market_agent import RealTimeMarketAgent

@pytest.fixture
def agent_with_fake_tavily(monkeypatch):
    """
    Fixture: RealTimeMarketAgent с замоканным методом tavily.search.
    Возвращает 3 фиктивные результата с названиями, url и publishedAt.
    """
    agent = RealTimeMarketAgent()
    fake_results = [
        {"title": "Headline A", "url": "http://a", "publishedAt": "2025-05-16T10:00:00Z"},
        {"title": "Headline B", "url": "http://b", "publishedAt": "2025-05-16T09:00:00Z"},
        {"title": "Headline C", "url": "http://c", "publishedAt": "2025-05-16T08:00:00Z"},
    ]
    # Подменяем tavily.search на фиктивную реализацию
    monkeypatch.setattr(agent.tavily, "search", lambda query: fake_results)
    return agent

def test_get_yahoo_finance_news(agent_with_fake_tavily):
    items = agent_with_fake_tavily.get_yahoo_finance_news(count=2)
    # Должно вернуть ровно 2 элемента
    assert len(items) == 2
    # Поля корректно переименованы и источник — Yahoo Finance
    assert items[0]["source"] == "Yahoo Finance"
    assert items[1]["source"] == "Yahoo Finance"
    # Заголовок первого элемента соответствует фиктивному
    assert items[0]["title"] == "Headline A"
    # URL берётся из fake_results
    assert items[1]["url"] == "http://b"
    # Дата сокращается до YYYY-MM-DD
    assert items[0]["publishedAt"] == "2025-05-16"

def test_get_cnbc_news(agent_with_fake_tavily):
    items = agent_with_fake_tavily.get_cnbc_news(count=1)
    assert len(items) == 1
    assert items[0]["source"] == "CNBC"
    assert items[0]["title"] == "Headline A"

def test_get_reuters_news(agent_with_fake_tavily):
    items = agent_with_fake_tavily.get_reuters_news(count=3)
    assert len(items) == 3
    assert items[2]["source"] == "Reuters"
    assert items[2]["title"] == "Headline C"
