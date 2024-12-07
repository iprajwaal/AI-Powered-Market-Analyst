import os
import yaml
from typing import Any
from typing import Dict
from src.config.logging import logger

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._config = cls._instance._load_config()
        return cls._instance
    
    def __init__(self, config_path: str = "None"):
        """
        Initialize the Config object with the path to the config file.

        Args:
            config_path (str, optional): Path to the YAML config file. Defaults to "None".
        """

        if self.__initialized:
            return
        self.__initialized = True

        config_path = config_path or self._find_config_path()
        logger.info(f"Loading config from {config_path}")

        self._config = self._load_config(config_path)
        if self._config:
            self.PROJECT_ID = self._config.get("project_id")
            self.REGION = self._config.get("region")
            self.CREDENTIALS_PATH = self._config.get("credentials_path")
            self.MODEL_NAME = self._config.get("model_name")

            if self.CREDENTIALS_PATH:
                self._set_google_credentials(self.CREDENTIALS_PATH)
            else:
                logger.warning("No credentials path provided in config.")
        else:
            logger.error("No config loaded. Please check the file path.")

    @staticmethod
    def _find_config_path() -> str:
        """
        Find the path to the config file.

        Returns:
            str: Path to the config file.
        """
        possible_paths = [
            os.path.abspath("./backend/config/config.yaml"),
            os.path.abspath("./config/config.yaml"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found config file at {path}")
                return path
            
        logger.error("No config file found in any default location.")
        raise FileNotFoundError("No config file found in any default location.")
    
    @staticmethod
    def _find_credentials_path() -> str:
        """
        Find the path to the credentials file.

        Returns:
            str: Path to the credentials file.
        """
        possible_paths = [
            os.path.abspath("./backend/config/key.json"),
            os.path.abspath("./config/key.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found credentials file at {path}")
                return path
            
        logger.error("No credentials file found in any default location.")
        raise FileNotFoundError("No credentials file found in any default location.")
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load the config file.

        Args:
            config_path (str): Path to the config file.

        Returns:
            Dict[str, Any]: The config file as a dictionary.
        """
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error loading config file: {e}")
            return None
        
    @staticmethod
    def _set_google_credentials(credentials_path: str) -> None:
        """
        Set the Google credentials environment variable.

        Args:
            credentials_path (str): Path to the credentials file.
        """
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            logger.info("Google credentials set from {credentials_path}")
        else:
            logger.error(f"Credentials file not found at {credentials_path} Google credentials not set.")

config = Config()