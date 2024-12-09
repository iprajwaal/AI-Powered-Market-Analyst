from serpapi import GoogleSearch
from kaggle.api.kaggle_api_extended import KaggleApi
import json
import logging
from src.config.setup import Config

config = Config()
logger = logging.getLogger(__name__)


def dataset_search(query: str) -> str:
    """Searches for datasets related to the given query."""
    try:
        search_query = f"{query} dataset"
        search = GoogleSearch({"engine": "google_datasets", "q": search_query, "api_key": config.SERPAPI_KEY})
        results = search.get_json()

        if "error" in results:
           return json.dumps({"error": results["error"]})

        google_datasets_results = format_dataset_results(results)

        try:
            api = KaggleApi()
            api.authenticate()
            kaggle_datasets = api.datasets_list(search=query) # Use Kaggle API to search datasets
            kaggle_results = format_kaggle_results(kaggle_datasets)
        except Exception as e:
            logger.error(f"Error using Kaggle API: {e}")  # Handle Kaggle errors
            kaggle_results = [] #Return empty list in case of error

        return json.dumps({
            "google_datasets_results": google_datasets_results, 
            "kaggle_results": kaggle_results
        }, indent=2)


    except Exception as e:
        logger.error(f"Error in dataset_search: {e}")
        return json.dumps({"error": str(e)})


def format_dataset_results(results: dict) -> list:
    """Formats dataset search results."""
    organic_results = results.get("organic_results", [])
    formatted_results = []
    for result in organic_results:
        try:
            formatted = {
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet"),
            }
            formatted_results.append(formatted)
        except Exception as e:
           logger.error(f"Error formatting dataset result: {e}")

    return formatted_results



def format_kaggle_results(datasets: list) -> list:
    """Formats Kaggle dataset search results."""
    formatted_results = []
    for dataset in datasets:
        try:
            formatted = {
                "title": dataset.title,
                "link": f"https://www.kaggle.com/datasets/{dataset.ref}",
                "snippet": dataset.snippet
            }
            formatted_results.append(formatted)
        except Exception as e:
            logger.error(f"Error formatting dataset result: {e}")

    return formatted_results


if __name__ == "__main__":
    test_queries = [
        "Cosmetics customer segmentation",
        "Financial time series", # Example
        "Image classification"  # Example
    ]
    for query in test_queries:
        results = dataset_search(query)
        print(f"Dataset Search Results for '{query}':\n{results}\n")