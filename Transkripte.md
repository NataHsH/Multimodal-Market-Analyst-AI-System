# 📑 Transkripte und Folien von Ergebnispräsentationen

## 🎓 Abschlussprojekt: „Multimodales Marktanalyse-KI-System“
Dieses Projekt wurde im Rahmen des Studiengangs an der Hochschule Hannover durchgeführt. Ziel war es, ein intelligentes Multi-Agenten-System zu entwickeln, das Investor-Relations-Dokumente (IR) nutzt, um Finanzdaten zu analysieren, zu prognostizieren und verständlich darzustellen.

---

## 🧭 Systemübersicht & Agentenrollen

### 🌟 1. Multimodaler RAG-Agent
- **Funktion:** Extraktion relevanter Finanzinformationen aus PDFs.
- **Typischer Prompt:**  
  *„Fassen Sie die jüngste Finanzleistung von NVIDIA basierend auf dieser Ergebnispräsentation zusammen.“*
- **Antwortbeispiel:**  
  *„NVIDIAs Umsatz im Q4 GJ24 stieg um 18 %, getrieben durch starke GPU-Verkäufe (Quelle: NVIDIA Q4 FY24 Earnings Slides, Seite 5).“*

---

### 🌟 2. Datenwissenschafts- & Analyse-Agent
- **Funktion:** Zeitreihenanalysen und Prognoseerstellung.
- **Typischer Prompt:**  
  *„Analysieren Sie Microsofts Aktienentwicklung im letzten Jahr und prognostizieren Sie die Performance im nächsten Quartal.“*
- **Antwortbeispiel:**  
  *„Microsoft zeigte 2023 ein moderates Wachstum. Die Prognose für Q1 2024 deutet auf eine leichte Steigerung hin (basierend auf extrapolierten IR-Werten).“*

---

### 🌟 3. Websuche- & Echtzeit-Markt-Agent
- **Funktion:** Abruf aktueller Marktinformationen und Stimmungsanalysen.
- **Typischer Prompt:**  
  *„Was sind die neuesten Nachrichten, die heute den Aktienkurs von Google beeinflussen?“*
- **Antwortbeispiel:**  
  *„Googles Aktie stieg heute um 3 %, ausgelöst durch positive Reaktionen auf neue KI-Produktankündigungen (Quelle: CNBC, Mai 2025).“*

---

### 🌟 4. Koordinator-Agent
- **Funktion:** Zentrale Steuerung, Aufgabenverteilung und Ergebnisaggregation.
- **Beispiel-Workflow:**  
  1. Anfrage empfangen  
  2. Aufgaben an Agenten delegieren  
  3. Ergebnisse kombinieren  
  4. Multimodale Antwort mit Quellenangabe erzeugen

---

## 🖥️ Systempräsentation

- Die Benutzeroberfläche wurde mit **Gradio** realisiert.
- Die Anwendung wurde auf **Hugging Face Spaces** bereitgestellt.
- Jede Agentenantwort enthält **Quellenangaben** aus IR-Dokumenten.
- **Zusatzaufgaben (z. B. QA-Agent)** wurden im Rahmen des Projekts **nicht umgesetzt**.

---

## 📦 Projektstatus

- ✅ Hauptfunktionen aller vier Agenten erfolgreich implementiert.
- ❌ Keine interaktiven Visualisierungen oder Prognose-Grafiken in der Endpräsentation.
- 📋 Ergebnispräsentation erfolgte ohne Zusatzmodule (z. B. Qualitätssicherung, ethische Prüfungen).

---

## 📁 Weiterführende Materialien

- Quellcode: *GitHub-Repository*
- Anwendung: *Gradio-Interface auf Hugging Face*
- Projektdokumentation: *Technischer Bericht*
- Agile Planung: *Jira-Board*

---

## 🧠 Fazit

Das System demonstriert erfolgreich die Fähigkeit, reale Finanzdokumente automatisiert auszuwerten. Es bietet:
- Relevante Informationen mit Quellenangaben
- Agentenbasierten Workflow
- Praktische Anwendung generativer KI für die Finanzbranche

