# Multimodal Market Analyst AI System 🌐

A multimodal AI system for financial analysis based on IR documents from Apple, Microsoft, Google, NVIDIA, and Meta.

---

## 🚀 Features

- Analysis of IR documents (PDFs, presentations, transcripts)
- Forecasting and visualizations
- Real-time market data and news
- Coordination of multiple agents for combined analysis
- Response quality monitoring (optional)

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ✅ Start the Application

```bash
python app.py
```

---

## 💡 Example Queries

1. **Earnings Presentation Analysis:**
   - Query: "Summarize the last quarterly performance of NVIDIA."
   - Expected Output: "NVIDIA's revenue increased by 18% in Q4 FY24 (Source: NVIDIA Q4 FY24 Earnings Slides, Page 5)."

2. **Forecast for Microsoft:**
   - Query: "Generate a forecast for Microsoft's stock price in the next quarter."
   - Expected Output: Visualization with Plotly and a forecast summary.

3. **Latest Market News:**
   - Query: "What are the latest news about Google?"
   - Expected Output: "Google's stock rose by 3% today due to positive AI product announcements (Source: CNBC, May 2025)."

---

## 📦 Data Sources

- IR documents from Apple, Microsoft, Google, NVIDIA, and Meta (2020–2024)
- Document Types:
  - Annual Reports (10-K)
  - Quarterly Reports (10-Q)
  - Earnings Presentations and Transcripts
  - Investor Presentations, Charts

---

## 🔥 Technologies

| Agent            | Tools/Models                                |
|------------------|---------------------------------------------|
| RAG Specialist   | CLIP, SentenceTransformers, Chroma, Gemini  |
| Analysis Agent   | Pandas, Matplotlib, Plotly, Prophet         |
| Web Search Agent | SerpAPI, Tavily, BeautifulSoup              |
| Coordinator      | LangChain, LangGraph                       |
| QA Agent (opt.)  | BERT, GPT-Moderation-API                   |

---

## 🎯 Workflow (Scenario)

1. **User Query:**
   - "Analyze Meta's stock performance over the last year and provide a forecast for the next quarter."

2. **Coordinator Agent:**
   - Splits the query into three subtasks:
     - Data retrieval from IR documents (RAG Specialist)
     - Forecasting based on historical data (Analysis Agent)
     - Real-time updates from web sources (Web Search Agent)

3. **Data Processing and Output:**
   - Aggregation of results and presentation through the Gradio interface.

---

## 📅 Timeline & Milestones

**Week 1:**
- Data collection and preprocessing
- Implementation of RAG and Analysis Agents
- Initial module testing

**Week 2:**
- Integration of Web Search Agent
- Coordinator Agent development
- Gradio interface development
- End-to-end testing and fine-tuning

---

## 🧑‍💻 User Guide

- Open the Gradio interface via the provided link.
- Enter the desired query (e.g., "Summarize the latest earnings presentation from Apple").
- Wait for all agents to complete their tasks.
- Review the results as text output and visualizations.

---

## 📑 License

This project is licensed under the MIT License. For more details, see the `LICENSE` file.

---

## 🎬 Demo & Presentation

- Run the Gradio app locally using `python app.py` or access the hosted version on Hugging Face Spaces.
- Example inputs and expected outputs are outlined in the `README.md`.
- For presentations: Follow the defined workflow and demonstrate the responses from each agent in real-time.

---

## ✅ Next Steps

- Expand the QA Agent for quality checks and ethical validation.
- Integrate additional data sources for real-time forecasting.
- Optimize the models by fine-tuning on financial datasets.

---
