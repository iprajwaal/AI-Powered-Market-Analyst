from src.config.log_config import logger
from src.utils.io import load_yaml
from serpapi import GoogleSearch
from src.config.setup import Config
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from typing import Tuple
import requests
import json

# Static paths
CREDENTIALS_PATH = '.backend/credentials/key.yml'

def google_search(query: str, engine: str = "google", num_results: int = 10) -> str:
    """
    Perform a Google search with the given query.

    Args:
        query (str): The query to search for.
        engine (str, optional): The search engine to use. Defaults to "google".
        num_results (int, optional): The number of results to return. Defaults to 10.

    Returns:
        str: The search results.
    """
    try:
        search = GoogleSearch({
            "q": query,
            "engine": engine,
            "num": num_results
        })
        results = search.get_dict()

        if "error" in results:  # Handle SerpAPI errors
            return json.dumps({"error": results["error"]})

        formatted_results = format_top_search_results(results)
        return json.dumps({"organic_results": formatted_results}, indent=2) #Return top organic results


    except Exception as e:
        logger.error(f"Error in google_search: {e}")
        return json.dumps({"error": str(e)})
    
def format_top_search_results(results: dict) -> list:
    """Formats top search results from SerpAPI."""
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
        except Exception as e:  # Handle potential errors during formatting
            logger.error(f"Error formatting result: {e}, Result: {result}")

    return formatted_results


if __name__ == "__main__":
    test_queries = [
        "Cosmetics industry market analysis",
        "Sephora competitor analysis",
        "Luxury handbag market trends"
    ]
    for query in test_queries:
        results = google_search(query)
        print(f"Results for '{query}':\n{results}\n")

