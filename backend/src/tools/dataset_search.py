from typing import Dict, List, Union
import json
import logging
import os
import shutil
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from src.config.setup import Config

logger = logging.getLogger(__name__)

def setup_kaggle_credentials(kaggle_creds_path: str) -> None:
    """Setup Kaggle credentials with proper validation"""
    try:
        if not os.path.exists(kaggle_creds_path):
            raise FileNotFoundError(f"Kaggle credentials not found at {kaggle_creds_path}")

        # Verify credentials format
        with open(kaggle_creds_path) as f:
            creds = json.load(f)
            if not all(k in creds for k in ['username', 'key']):
                raise ValueError("Invalid credentials format - missing username or key")

        # Setup target directory
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        # Copy and set permissions
        target_path = kaggle_dir / 'kaggle.json'
        shutil.copy2(kaggle_creds_path, target_path)
        os.chmod(target_path, 0o600)
        
        logger.info("Kaggle credentials setup successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup Kaggle credentials: {e}")
        raise

def initialize_kaggle_api() -> KaggleApi:
    """Initialize and authenticate Kaggle API"""
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        logger.error(f"Failed to initialize Kaggle API: {e}")
        raise

def dataset_search(query: str) -> str:
    """Searches for datasets related to the given query."""
    try:
        api = initialize_kaggle_api()
        datasets = api.dataset_list(search=query, sort_by='relevance')
        
        results = []
        for dataset in datasets[:5]:
            results.append({
                'name': dataset.ref,
                'title': dataset.title,
                'size': dataset.size,
                'url': f"https://www.kaggle.com/{dataset.ref}",
                'description': dataset.description
            })
            
        formatted_results = {
            'query': query,
            'datasets': results,
            'total_found': len(results)
        }
        
        return json.dumps(formatted_results, indent=2)
        
    except Exception as e:
        error_msg = f"Error searching datasets: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    kaggle_creds_path = "/Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/credentials/kaggle.json"
    setup_kaggle_credentials(kaggle_creds_path)

    test_queries = [
        "Cosmetics customer segmentation",
        "Financial time series",
        "Image classification"
    ]
    
    for query in test_queries:
        results = dataset_search(query)
        print(f"Dataset Search Results for '{query}':\n{results}\n")