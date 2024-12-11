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