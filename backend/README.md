### Installation

1. Clone the repository:
   ```
   git clone https://github.com/iprajwaal/AI-Powered-Market-Analyst.git
   cd AI-Powered-Market-Analyst/backend
   ```

2. Set up a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```
3. **Install Dependencies** 📦

    ```bash
    pip install -r requirements.txt
    ```
4. Install Poetry (if not already installed):
   ```
   pip install poetry
   ```

5. Install project dependencies:
   ```
   poetry install
   ```

6. Set up environment variables:
   ```
   export PYTHONDONTWRITEBYTECODE=1
   export PYTHONPATH=$PYTHONPATH:./backend
   ```

### Setting up Credentials

This project uses API keys for Google Cloud Platform (GCP) and SERP API. Store these securely.  Ensure you have created a `credentials` directory in your project.

1. **Create a `credentials` folder in the project root:**

   ```
   mkdir credentials
   ```

2. **GCP Service Account Credentials:**

- Go to the [Google Cloud Console](https://console.cloud.google.com/).
- Create a new project or select an existing one.
- Navigate to *APIs & Services* > *Credentials*.
- Click *Create Credentials* > *Service Account Key*.
- Select your service account, choose JSON as the key type, and click *Create*.
- Save the downloaded JSON file as `key.json` inside the `credentials` folder.

3. **SERP API Credentials:**

- Sign up for a [SERP API](https://serpapi.com/) account.
- Obtain your API key from the dashboard.
- Create a file named `key.yml` inside the `credentials` folder.
- Add your SERP API token:

   ```yaml
   serp:
     key: your_serp_api_key_here
    ```



# AI Market Analyst

AI Market Analyst is a multi-agent system that uses advanced LLMs to research companies and industries, generate tailored AI use cases, and provide implementation resources. The system leverages DSPy, LangChain, and LangGraph to create a sophisticated analysis pipeline with web-grounded information.

## Features

- **Industry & Company Research**: Automatically researches companies and industries using web search
- **AI Use Case Generation**: Creates tailored AI/ML implementation strategies with business value analysis
- **Resource Discovery**: Finds relevant datasets and implementation resources from Kaggle, HuggingFace, and GitHub
- **Structured Reasoning**: Uses DSPy for enhanced chain-of-thought reasoning and structured outputs
- **Full-Stack Implementation**: Includes a FastAPI backend and React/Next.js frontend


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

   # LLM Configuration
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-1.5-pro
   LLM_TEMPERATURE=0.2
   LLM_MAX_TOKENS=8192

   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   API_DEBUG=True
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
3. Submit the form and wait for the analysis to complete
4. Explore the generated use cases and resources
5. Download the markdown report if desired

## API Endpoints

- `POST /api/analyze`: Start an analysis with company or industry information
- `GET /api/analysis/{request_id}`: Get the status and results of an analysis
- `GET /api/markdown/{request_id}`: Get the markdown output for an analysis
- `GET /health`: Health check endpoint

## Technical Details

### Backend Components

- **FastAPI**: Web framework for the API
- **DSPy**: Framework for structured prompting and chain-of-thought reasoning
- **LangChain**: Framework for tool integration
- **LangGraph**: Framework for workflow orchestration
- **Gemini API**: Large language model for text generation
- **SerpAPI**: Web search API for information retrieval
- **Kaggle API**: Dataset discovery

### Frontend Components

- **Next.js**: React framework for the web application
- **Tailwind CSS**: Utility-first CSS framework for styling
- **TypeScript**: Type-safe JavaScript

## License

[MIT](LICENSE)

## Acknowledgements

- [DSPy](https://github.com/stanfordnlp/dspy) for structured prompting
- [LangChain](https://github.com/langchain-ai/langchain) for LLM applications
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration