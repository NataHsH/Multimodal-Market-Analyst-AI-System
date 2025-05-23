# 🚀 Multimodales Marktanalyse-KI-System

🔗 **Live Demo on Hugging Face Spaces:** [https://huggingface.co/spaces/vladimir707/Multimodal_Market_Analyst_AI_System](https://huggingface.co/spaces/vladimir707/Multimodal_Market_Analyst_AI_System)

Dieses Projekt ist ein modulares, agentenbasiertes KI-System zur Analyse von Finanzmarktdaten unter Verwendung multimodaler Eingaben wie Investorenberichte, Diagramme und Echtzeitnachrichten. Es wurde im Rahmen eines Hochschul-Abschlussprojekts entwickelt, um fortgeschrittene Techniken zur Datengewinnung, -analyse und -visualisierung in finanziellen Kontexten zu demonstrieren.

---

## 📦 Projektstruktur

Das System besteht aus vier spezialisierten KI-Agenten, die von einem zentralen Koordinator-Agenten gesteuert werden:

### 1. 🧠 Multimodaler RAG-Agent
- **Zweck:** Extrahiert relevante Informationen aus IR-Dokumenten (PDFs) mithilfe von Vektorsuche und metadatenbasiertem Indexing.
- **Technologien:** LangChain, pdfplumber, ChromaDB, SentenceTransformers, Google Gemini
- **Funktionen:**
  - Analysiert Finanzdokumente und indexiert sowohl Text als auch Tabellen.
  - Gibt Zitate mit Firma, Jahr, Formularart und Seitenreferenzen aus.
  - Betten und speichert Dokumentteile zur späteren Abfrage.

### 2. 📈 Datenwissenschafts- & Prognoseagent (DS-Agent)
- **Zweck:** Führt Zeitreihenanalysen durch und erstellt einfache lineare Prognosen.
- **Technologien:** LangChain, matplotlib, reguläre Ausdrücke, Google Gemini
- **Funktionen:**
  - Extrahiert numerische Werte aus jährlichen Textdaten.
  - Führt delta-basierte Extrapolation für zukünftige Werte durch.
  - Erstellt und speichert Prognosediagramme.

### 3. 🌍 Echtzeit-Markt-Agent
- **Zweck:** Sammelt aktuelle Finanznachrichten, Aktienkurse, Intraday-Daten und führt Sentimentanalysen durch.
- **Technologien:** yFinance, Alpha Vantage, NewsAPI, Tavily, HuggingFace Transformers, LangChain
- **Funktionen:**
  - Sammelt Nachrichten von Yahoo Finance, CNBC, Reuters.
  - Ruft Kursänderungen und Handelsvolumen ab.
  - Führt mehrsprachige Sentimentanalyse durch.
  - Gibt BUY/SELL/HOLD-Empfehlungen auf Basis der Daten.

### 4. 🧭 Koordinator-Agent
- **Zweck:** Koordiniert und delegiert Aufgaben an die entsprechenden Spezialagenten.
- **Technologien:** LangChain, Gradio
- **Funktionen:**
  - Bietet eine einheitliche Chat-Oberfläche via Gradio.
  - Steuert den Interaktionsfluss und erzeugt multimodale Antworten.
  - Behandelt Fehlerfälle und Ausweichstrategien.

---

## 🛠️ Technologiestack

- **Sprache:** Python  
- **Frameworks/Bibliotheken:** LangChain, Google Gemini, Gradio, Matplotlib, HuggingFace, pdfplumber, yfinance, ChromaDB  
- **Genutzte APIs:** Google Generative AI, Tavily, Alpha Vantage, NewsAPI  

---

## 🗂️ Datensatz

Das System verwendet IR-Dokumente (2020–2024) von:

- Apple  
- Microsoft  
- Google (Alphabet)  
- NVIDIA  
- Meta  

Dokumentarten:

- Jahres- und Quartalsberichte (10-K, 10-Q)  
- Präsentationsfolien  
- Transkripte von Earnings Calls  
- Finanzielle Tabellen und Diagramme  

---

## 🧪 Entwicklungsablauf

### Woche 1:
- Aufbereitung und Bereinigung der IR-Dokumente (PDF)  
- Erstellung und Test des multimodalen Dokumenten-Extractors und Embedders  
- Implementierung von ChromaDB zur Speicherung und Abfrage  
- Entwicklung des Prognosetools für Finanzdaten  

### Woche 2:
- Integration von Echtzeit-APIs und Sentimentmodellen  
- Entwicklung des zentralen Koordinator-Agenten  
- Bereitstellung des Prototyps mit Gradio  
- Finalisierung der Agentenkoordination und Ein-/Ausgabelogik  

---

## 🖼️ Beispielhafte Nutzeranfrage

> „Wie hoch sind Apples Nettoumsätze in 2022, 2023 und 2024? Bitte eine Prognose erstellen.“

Systemantwort:

- Daten aus den ursprünglichen IR-Berichten extrahiert  
- Prognose als PNG-Diagramm visualisiert  
- Quellenangabe enthalten: „Apple, 2024, 10-K, Seite 12“  

---

## 🧪 Ausführung

1. Abhängigkeiten installieren:  
   ```bash
   pip install -r requirements.txt

2. Umgebung einrichten:
    .env muss API-Schlüssel für NewsAPI, Tavily, Alpha Vantage, Google enthalten.

3. UI starten:
    python app.py

---

## 📊 Ausgabe

    - Strukturierte Antwort mit Quellen (RAG)

    - Finanzprognose mit Diagramm (DS-Agent)

    - Echtzeitanalyse inkl. Sentiment (Web-Agent)

---

## 📍 Autoren

Projektteam an der Hochschule Hannover

    - Betreuer: Hussam Alafandi

    - Teammitglieder:

        - Volodymyr Kyryliuk – Design des Multimodal-Agenten

        - Nataliia Honcharova – Datenanalyse & Prognose

        - Volodymyr Tymoshchuk – Integration von Echtzeitdaten

---

## 📃 Lizenz

Dieses Projekt wurde im Rahmen einer Hochschulveranstaltung an der Hochschule Hannover entwickelt.

Lizenziert unter der MIT-Lizenz.
