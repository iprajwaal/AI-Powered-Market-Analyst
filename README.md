# 💹 AI-Powered Market Analyst

> The Automated Market Researcher uses AI to streamline market research and generate potential use cases. This tool helps businesses quickly identify market trends, analyze competitor strategies, and discover new opportunities.



![Project Overview](https://github.com/user-attachments/assets/c8b693af-7555-4884-89bb-203bf9517dfa)

## 🛠️ Project Structure

```plaintext
AI-Powered-Market-Analyst/
├── backend/
│   ├── config/
│   │   └── config.yml : Configuration setup for Google Cloud, Vertex AI, and Kaggle API.
│   ├── credentials/
│   │   ├── key.json : Google Cloud Platform service account key.
│   │   ├── api.yml : SERP API key.
│   │   └── kaggle.json : Kaggle API key.
│   ├── data/
│   │   ├── react.txt : Stores input prompts for the ReAct agent.
│   │   └── output/
│   │       └── trace.json : Output traces from example runs.
│   ├── src/
│   │   ├── config/
│   │   │   └── setup.py : Configuration setup for Google Cloud, Vertex AI, and Kaggle API.
│   │   │   └── log_config.py : Logging configuration.
│   │   ├── llm/
│   │   │   └── gemini.py : Implementation of the Gemini model.
│   │   ├── react/
│   │   │   ├── agent.py : Core ReAct agent implementation.
│   │   ├── tools/
│   │   │   ├── google_search.py : Google Search via SERP API.
│   │   │   ├── dataset_search.py : Kaggle dataset search.
│   │   │   ├── competitor_analysis.py : Competitor analysis tool.
│   │   │   └── industry_report.py : Industry report generation.
│   │   │   └── manager.py : Manages tool selection and execution.
│   │   └── utils/
│   │       └── io.py : Input/output utilities.
├── client/
└── requirements.txt
```

## ⏳ Agentic workflow


![Agentic flow](https://github.com/user-attachments/assets/940f3973-5b36-4af9-bc39-f501c53afc32)
