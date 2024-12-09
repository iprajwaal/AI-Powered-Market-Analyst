from serpapi import GoogleSearch
from src.config.setup import Config
import json
import logging

config = Config()
logger = logging.getLogger(__name__)

def product_search(query: str, use_google_shopping: bool = True) -> str:
    """
    Searches for products.  Uses Google Shopping if available, otherwise falls back to standard Google Search.
    """
    try:
        if use_google_shopping:
            try:
                search = GoogleSearch({"engine": "google_product", "product_id": query, "api_key": config.SERPAPI_KEY})  # Use product ID or other identifier
                results = search.get_json()
                if "error" in results:
                    raise ValueError(results["error"]) 
                return json.dumps(results, indent=2)

            except ValueError as e:
                logger.warning(f"Google Shopping search failed: {e}. Falling back to standard Google Search.")


        search_query = f"{query} product"
        search = GoogleSearch({"q": search_query, "api_key": config.SERPAPI_KEY, "num": 10})  # You might increase num.
        results = search.get_json()

        if "error" in results:
            return json.dumps({"error": results["error"]})


        organic_results = results.get("organic_results", [])
        return json.dumps({"organic_results": organic_results}, indent=2) # Use 'organic_results" consistently


    except Exception as e:
        logger.error(f"Error in product_search: {e}")
        return json.dumps({"error": str(e)})  # Return JSON error