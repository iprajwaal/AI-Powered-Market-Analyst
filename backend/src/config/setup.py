import os
import yaml
import google.auth
import google.auth.transport.requests
import requests
import logging
from typing import Any, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = "/Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/config/config.yml"):
        """
        Initialize the Config class.

        Args:
        - config_path (str): Path to the YAML configuration file.
        """
        if self.__initialized:
            return
        self.__initialized = True
        
        self.__config = self._load_config(config_path)
        self.PROJECT_ID = self.__config['project_id']
        self.REGION = self.__config['region']
        self.CREDENTIALS_PATH = self.__config['credentials_json']
        self._set_google_credentials(self.CREDENTIALS_PATH)
        self.MODEL_NAME = self.__config['model_name']
        self.API_KEY = self._load_api_key()
        self.setup_client()
        self._initialize_kaggle()

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load the YAML configuration from the given path.

        Args:
        - config_path (str): Path to the YAML configuration file.

        Returns:
        - dict: Loaded configuration data.
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load the configuration file. Error: {e}")

    @staticmethod
    def _set_google_credentials(credentials_path: str) -> None:
        """
        Set the Google application credentials environment variable.

        Args:
        - credentials_path (str): Path to the Google credentials file.
        """
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    def _load_api_key(self):
        """
        Load the API key from the api.yml file.

        Returns:
        - str: The API key.
        """
        api_key_path = "/Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/credentials/api.yml"
        try:
            with open(api_key_path, 'r') as file:
                api_key_data = yaml.safe_load(file)
                return api_key_data['api_key']
        except Exception as e:
            logger.error(f"Failed to load the API key. Error: {e}")
            return None

    def setup_client(self):
        """
        Set up the client for making requests to the Generative Language API.
        """
        self.client = requests.Session()
        self.client.headers['Authorization'] = f'Bearer {self.API_KEY}'

    def _initialize_kaggle(self) -> None:
        """Initialize Kaggle credentials"""
        try:
            kaggle_dir = os.path.expanduser('~/.kaggle')
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_target = os.path.join(kaggle_dir, 'kaggle.json')
            
            import shutil
            shutil.copy2(self.__config['kaggle_credentials'], kaggle_target)
            
            os.chmod(kaggle_target, 0o600)
            logger.info("Successfully initialized Kaggle credentials")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kaggle credentials: {e}")
            raise

    def generate_text(self, prompt: str, max_retries=3, backoff_factor=0.5) -> str:
        """
        Generate text using the Generative Language API with retry mechanism.

        Args:
        - prompt (str): The prompt for text generation.
        - max_retries (int): Maximum number of retry attempts.
        - backoff_factor (float): Factor for exponential backoff between retries.

        Returns:
        - str: The generated text.
        """
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL_NAME}:generateContent"
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        # Create a retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        # Mount the adapter to the session
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.client.mount("https://", adapter)
        
        try:
            response = self.client.post(url, json=data, timeout=30)  # Added timeout
            if response.status_code == 200:
                return response.json()['contents'][0]['parts'][0]['text']
            else:
                error_msg = f"Error generating text: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"API Error: {response.status_code}"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error: {e}")
            return "Connection error occurred. Please check your network or API endpoint."
        except requests.exceptions.Timeout as e:
            logger.error(f"Request timed out: {e}")
            return "Request timed out. The API server might be overloaded."
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"Unexpected error: {str(e)}"

config = Config()

# Example usage
prompt = "Hello, world!"
generated_text = config.generate_text(prompt)
print(generated_text)
