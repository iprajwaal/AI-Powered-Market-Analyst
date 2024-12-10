import os
import yaml
import google.auth
import google.auth.transport.requests
import openai
import logging
from typing import Any, Dict

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
        self.setup_openai_client()

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

    def setup_openai_client(self):
        """
        Set up the OpenAI client using the configuration and Google credentials.
        """
        scopes = [
        'https://www.googleapis.com/auth/cloud-platform',
        ]
        creds, project = google.auth.default(scopes=scopes)  # Use scopes here!


        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req) 

        PROJECT = self.__config['project_id']
        ENDPOINT_NAME = self.__config['endpoint_name']

        # Load API key from api.yml
        api_key_path = "/Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/credentials/api.yml"
        with open(api_key_path, 'r') as file:
            api_key_data = yaml.safe_load(file)
            api_key = api_key_data['api_key']

        self.client = openai.OpenAI(
        api_key=api_key,
        base_url=f"https://{config.REGION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}/locations/{config.REGION}/endpoints/{ENDPOINT_NAME}"
        )


config = Config()

# Example usage of the client
response = config.client.Completions.create(
    model=config.__config['model_name'],
    prompt="Hello, world!",
    max_tokens=5
)

print(response)