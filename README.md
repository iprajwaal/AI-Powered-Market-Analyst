# 💹 AI-Powered Market Analyst

> AI Market Analyst is a multi-agent system that uses advanced LLMs to research companies and industries, generate tailored AI use cases, and provide implementation resources. The system leverages LangChain, and LangGraph to create a sophisticated analysis pipeline with web-grounded information.

![Project Overview](https://github.com/user-attachments/assets/c8b693af-7555-4884-89bb-203bf9517dfa)

## Features

- **Industry & Company Research**: Automatically researches companies and industries using web search
- **AI Use Case Generation**: Creates tailored AI/ML implementation strategies with business value analysis
- **Resource Discovery**: Finds relevant datasets and implementation resources from Kaggle, HuggingFace, and GitHub
- **Structured Reasoning**: Uses DSPy for enhanced chain-of-thought reasoning and structured outputs
- **Full-Stack Implementation**: Includes a FastAPI backend and React/Next.js frontend

## 🛠️ Project Structure

```plaintext
AI-Powered-Market-Analyst/
├── backend/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configuration for APIs and environments
│   │   └── logging_config.py    # Logging setup
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── industry_research_agent.py
│   │   ├── use_case_generation_agent.py
│   │   ├── resource_collection_agent.py
│   │   └── orchestrator.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_tools.py
│   │   ├── dataset_tools.py
│   │   ├── analysis_tools.py
│   │   └── document_tools.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── llm_interface.py
│   │   └── gemini_client.py
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── research_workflow.py
│   │   ├── use_case_workflow.py
│   │   └── resource_workflow.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py
│   │   ├── output_formatter.py
│   │   └── validation.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── langfuse_tracker.py
│   │   └── metrics.py
│   ├── app.py            # FastAPI application
│   └── requirements.txt  # Backend dependencies
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   ├── Header.jsx
│   │   │   │   ├── Footer.jsx
│   │   │   │   └── Layout.jsx
│   │   │   ├── ui/
│   │   │   │   ├── Button.jsx
│   │   │   │   ├── Card.jsx
│   │   │   │   ├── Input.jsx
│   │   │   │   └── Loader.jsx
│   │   │   ├── forms/
│   │   │   │   └── AnalysisForm.jsx
│   │   │   └── results/
│   │   │       ├── ResultsPanel.jsx
│   │   │       ├── UseCaseCard.jsx
│   │   │       └── ResourceLinks.jsx
│   │   ├── pages/
│   │   │   ├── index.js
│   │   │   ├── api/
│   │   │   │   └── hello.js
│   │   │   └── results/[id].js
│   │   ├── services/
│   │   │   └── api.js
│   │   └── styles/
│   │       └── globals.css
│   ├── package.json
│   └── tsconfig.json
│
└── README.md
```

## ⏳ Agentic workflow

![Agentic flow](https://github.com/user-attachments/assets/940f3973-5b36-4af9-bc39-f501c53afc32)

## Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Google API key for Gemini
- SERP API key for web search
- Kaggle API credentials (optional)
- Google Cloud credentials (optional, for VertexAI)

## Installation

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/iprajwaal/AI-Powered-Market-Analyst.git
   cd AI-Powered-Market-Analyst
   ```

2. Set up a virtual environment:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   export PYTHONDONTWRITEBYTECODE=1
   export PYTHONPATH=$PYTHONPATH:./backend
   ```

### Setting up Credentials

This project uses API keys for Google Cloud Platform (GCP), Gemini, and SERP API. Store these securely.

1. **Create a `credentials` folder in the project root:**
   ```bash
   mkdir credentials
   ```

2. **Gemini API Key:**
   - Visit the [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Save this key in your `.env` file

3. **SERP API Credentials:**
   - Sign up for a [SERP API](https://serpapi.com/) account
   - Obtain your API key from the dashboard
   - Create a file named `api.yml` inside the `credentials` folder:
     ```yaml
     serp:
       key: your_serp_api_key_here
     ```

4. **Kaggle API Credentials (Optional):**
   - Create a Kaggle account if you don't have one
   - Go to your account settings and create a new API token
   - Download the `kaggle.json` file and place it in the `credentials` folder

5. **Create `.env` file:**
    ```
    # API Keys
    GEMINI_API_KEY=your_gemini_api_key
    SERPAPI_KEY=your_serpapi_key
    KAGGLE_USERNAME=your_kaggle_username
    KAGGLE_KEY=your_kaggle_key
    HUGGINGFACE_API_KEY=your_huggingface_api_key

    # Optional Langfuse Monitoring
    LANGFUSE_PUBLIC_KEY=pk-lf-your_langfuse_public_key
    LANGFUSE_SECRET_KEY=sk-lf-your_langfuse_secret_key

    # LLM Configuration
    LLM_PROVIDER=llm_provider
    LLM_MODEL=llm_model
    ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

## Running the Application

### Start the Backend

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Frontend

```bash
cd frontend
npm run dev
```

The application will be available at http://localhost:3000.

## Usage

1. Navigate to the home page
2. Enter a company name (e.g., "Microsoft") or industry name (e.g., "Healthcare")
3. Set the Number of Use Cases to Generate
4. Submit the form and wait for the analysis to complete
5. Explore the generated use cases and resources
6. Download the markdown report if desired

## API Endpoints

- `POST /api/analyze`: Start an analysis with company or industry information
- `GET /api/analysis/{request_id}`: Get the status and results of an analysis
- `GET /api/markdown/{request_id}`: Get the markdown output for an analysis
- `GET /health`: Health check endpoint

## Technical Details

### Backend Components

- **FastAPI**: Web framework for the API
- **LangChain**: Framework for tool integration
- **LangGraph**: Framework for workflow orchestration
- **Gemini API**: Large language model for text generation
- **SerpAPI**: Web search API for information retrieval
- **Kaggle API**: Dataset discovery

### Frontend Components

- **Next.js**: React framework for the web application
- **Tailwind CSS**: Utility-first CSS framework for styling

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them
4. Push your changes to your fork
5. Create a pull request to the main repository

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for LLM applications
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
