# AI-Powered Market Analyst

> The Automated Market Researcher uses AI to streamline market research and generate potential use cases. This tool helps businesses quickly identify market trends, analyze competitor strategies, and discover new opportunities.



![Project Overview](https://github.com/user-attachments/assets/c8b693af-7555-4884-89bb-203bf9517dfa)

## Project Structure

### data/react.txt
Stores input prompts for the ReAct agent.

### data/output/
Contains output traces from example runs.


### backend/src/config/
Contains configuration setup and initialization for Google Cloud, Vertex AI, and Kaggle API.

### backend/src/tools/
Contains implementations for various tools used in market research:
- `google_search.py`: Google Search via SERP API
- `dataset_search.py`: Kaggle dataset search
- `competitor_analysis.py`: Competitor analysis tool
- `industry_report.py`: Industry report generation
- `manager.py`: Manages tool selection and execution


### backend/src/react/
The core ReAct agent implementation:
- `agent.py`: Main agent logic


### backend/credentials/
Stores credentials for accessing various APIs:
- `key.json`: Google Cloud credentials
- `api.yml`: API keys for external services
- `kaggle.json`: Kaggle API credentials
- `key.yml`: SerpAPI credentials


## Agentic workflow


![Agentic flow](https://github.com/user-attachments/assets/940f3973-5b36-4af9-bc39-f501c53afc32)
