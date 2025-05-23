# 🚀 Multimodal Market Analysis AI System

This project is a modular, agent-based AI system for analyzing financial market data using multimodal inputs such as investor reports, charts, and real-time news. It was developed as part of a university Abschlussprojekt (final project) to showcase advanced data retrieval, analysis, and visualization techniques in financial contexts.

---

## 📦 Project Structure

The system is composed of four specialized AI agents, coordinated by a central supervisor agent:

### 1. 🧠 Multimodal RAG Agent
- **Purpose:** Extracts relevant information from IR documents (PDFs) using vector similarity search and metadata-aware indexing.
- **Technologies:** LangChain, pdfplumber, ChromaDB, SentenceTransformers, Google Gemini
- **Functions:**
  - Parses financial documents and indexes both text and tables.
  - Retrieves citations with company, year, form type, and page references.
  - Embeds and stores document chunks for retrieval.

### 2. 📈 Data Science & Forecasting Agent (DS Agent)
- **Purpose:** Performs time series analysis and generates simple linear forecasts.
- **Technologies:** LangChain, matplotlib, regular expressions, Google Gemini
- **Functions:**
  - Parses numeric values from yearly textual data.
  - Performs delta-based extrapolation for future value.
  - Generates and saves forecast plots.

### 3. 🌍 Real-Time Market Agent
- **Purpose:** Gathers current financial news, stock prices, intraday data, and performs sentiment analysis.
- **Technologies:** yFinance, Alpha Vantage, NewsAPI, Tavily, HuggingFace Transformers, LangChain
- **Functions:**
  - Collects news from Yahoo Finance, CNBC, Reuters.
  - Retrieves price changes and volume activity.
  - Performs multilingual sentiment analysis.
  - Suggests BUY/SELL/HOLD signals based on data.

### 4. 🧭 Coordinator Agent
- **Purpose:** Orchestrates and delegates tasks to the appropriate specialized agents.
- **Technologies:** LangChain, Gradio
- **Functions:**
  - Exposes a unified chat interface via Gradio.
  - Manages interaction flow and composes multimodal responses.
  - Handles fallback and error reporting.

---

## 🛠️ Technology Stack

- **Language:** Python
- **Frameworks/Libraries:** LangChain, Google Gemini, Gradio, Matplotlib, HuggingFace, pdfplumber, yfinance, ChromaDB
- **APIs used:** Google Generative AI, Tavily, Alpha Vantage, NewsAPI

---

## 🗂️ Dataset

The system uses investor relations documents (2020–2024) from:

- Apple
- Microsoft
- Google (Alphabet)
- NVIDIA
- Meta

Document types include:

- Annual & Quarterly Reports (10-K, 10-Q)
- Presentation slides
- Transcripts from earnings calls
- Financial tables and charts

---

## 🧪 Development Workflow

### Week 1:
- Setup and cleaning of IR documents (PDF)
- Created and tested multimodal document extractor and embedder
- Implemented ChromaDB storage and retrieval
- Implemented data science forecasting tool

### Week 2:
- Integrated real-time APIs and sentiment models
- Built the central coordinator agent
- Deployed Gradio prototype
- Finalized agent coordination and unified I/O logic

---

## 🖼️ Sample User Query

> “What are Apple’s net sales in 2022, 2023, and 2024? Generate a forecast.”

System response:

- Retrieved data from original IR reports
- Visualized forecast as a PNG chart
- Included source citations: “Apple, 2024, 10-K, page 12”

---

## 🧪 How to Run

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt

2. Setup environment:
    .env must contain API keys for NewsAPI, Tavily, Alpha Vantage, Google.

3. Launch UI:
    python app.py
    
---

## 📊 Output

    - Structured answer with sources (RAG)

    - Financial forecast with chart (DS Agent)

    - Real-time summary + sentiment (Web Agent)
        
---

## 📍 Authors

Project team at Hochschule Hannover

   - Supervisor: Hussam Alafandi
   
   - Team members:

        - Volodymyr Kyryliuk – Multimodal Agent Design

        - Nataliia Honcharova – Data Science & Forecasting  

        - Volodymyr Tymoshchuk – Real-Time Data Integration

---

## 📃 License

This project was developed at Hochschule Hannover as part of an academic course.

Licensed under the [MIT License](LICENSE).
