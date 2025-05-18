# Multimodal Market Analyst AI System 🌐

Ein multimodales KI-System zur Finanzanalyse auf Basis von IR-Dokumenten von Apple, Microsoft, Google, NVIDIA und Meta.

---

## 🚀 Funktionen

- Analyse von IR-Dokumenten (PDFs, Präsentationen, Transkripte)
- Prognosen und Visualisierungen
- Echtzeit-Marktdaten und Nachrichten
- Koordination mehrerer Agenten zur kombinierten Analyse
- Überwachung der Antwortqualität (optional)

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ✅ Starten der Anwendung

```bash
python app.py
```

---

## 💡 Beispielanfragen

1. **Analyse einer Ergebnispräsentation:**
   - Anfrage: "Fassen Sie die letzte Quartalsleistung von NVIDIA zusammen."
   - Erwartete Ausgabe: "NVIDIAs Umsatz stieg im Q4 GJ24 um 18 % (Quelle: NVIDIA Q4 FY24 Earnings Slides, Seite 5)."

2. **Prognose für Microsoft:**
   - Anfrage: "Erstellen Sie eine Prognose für Microsofts Aktienkurs im nächsten Quartal."
   - Erwartete Ausgabe: Visualisierung mit Plotly und Prognosetext.

3. **Aktuelle Marktnachrichten:**
   - Anfrage: "Was sind die neuesten Nachrichten zu Google?"
   - Erwartete Ausgabe: "Googles Aktie stieg heute um 3 % aufgrund von positiven KI-Produktankündigungen (Quelle: CNBC, Mai 2025)."

---

## 📦 Datenquellen

- IR-Dokumente von Apple, Microsoft, Google, NVIDIA und Meta (2020–2024)
- Dokumenttypen:
  - Jahresberichte (10-K)
  - Quartalsberichte (10-Q)
  - Ergebnispräsentationen und Transkripte
  - Investorenpräsentationen, Diagramme

---

## 🔥 Technologien

| Agent            | Tools/Modelle                                |
|------------------|---------------------------------------------|
| RAG-Agent        | CLIP, SentenceTransformers, Chroma, Gemini  |
| Analyse-Agent    | Pandas, Matplotlib, Plotly, Prophet         |
| Websuche-Agent   | SerpAPI, Tavily, BeautifulSoup              |
| Koordinator      | LangChain, LangGraph                       |
| QA-Agent (opt.)  | BERT, GPT-Moderation-API                   |

---

## 🎯 Workflow (Szenario)

1. **Anfrage des Nutzers:**
   - "Analysieren Sie Metas Aktienkurs im letzten Jahr und prognostizieren Sie die Entwicklung im nächsten Quartal."

2. **Koordinator-Agent:**
   - Zerlegt die Anfrage in drei Teilaufgaben:
     - Datenabruf aus IR-Dokumenten (RAG-Agent)
     - Prognose basierend auf historischen Daten (Analyse-Agent)
     - Echtzeitinformationen aus Webquellen (Websuche-Agent)

3. **Datenverarbeitung und Ausgabe:**
   - Zusammenführung der Ergebnisse und Präsentation im Gradio-Interface.

---

## 📅 Zeitplan & Meilensteine

**Woche 1:**
- Datensammlung und Vorverarbeitung
- Implementierung von RAG- und Analyse-Agenten
- Erste Tests der Module

**Woche 2:**
- Integration von Websuche-Agent
- Aufbau des Koordinator-Agenten
- Entwicklung des Gradio-Interfaces
- End-to-End-Tests und Fine-Tuning

---

## 🧑‍💻 Benutzeranleitung

- Öffnen Sie das Gradio-Interface über den bereitgestellten Link.
- Geben Sie die gewünschte Analyseanfrage ein (z. B. "Fassen Sie die letzte Ergebnispräsentation von Apple zusammen").
- Warten Sie, bis alle Agenten ihre Aufgaben abgeschlossen haben.
- Überprüfen Sie die Ergebnisse in Textform und als Visualisierungen.

---

## 📑 Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Weitere Details finden Sie in der `LICENSE`-Datei.

---

## 🎬 Demo und Präsentation

- Starten Sie die Gradio-App lokal mit `python app.py` oder greifen Sie auf die gehostete Version auf Hugging Face Spaces zu.
- Beispieleingaben und erwartete Ergebnisse finden Sie in der `README.md`.
- Für Präsentationen: Folgen Sie dem definierten Workflow und demonstrieren Sie die Antworten der einzelnen Agenten in Echtzeit.

---

## ✅ Nächste Schritte

- Erweiterung des QA-Agenten für Qualitätsüberprüfung und ethische Validierung.
- Integration von weiteren Datenquellen für Echtzeitprognosen.
- Optimierung der Modelle durch Fine-Tuning auf Finanzdatensätzen.

---

