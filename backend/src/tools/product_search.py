from src.config.log_config import logger
from src.utils.io import load_yaml
from serpapi import GoogleSearch
import json

# Static paths
CREDENTIALS_PATH = '/Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/credentials/key.yml'

def product_search(query: str, product_id: str = None, use_google_shopping: bool = True) -> str:
    """
    Searches for products. Uses Google Shopping if product_id is provided; otherwise, falls back to standard Google Search.

    Args:
        query (str): The product name.
        product_id (str, optional): The specific product ID for Google Shopping. Defaults to None.
        use_google_shopping (bool, optional): Whether to use Google Shopping. Defaults to True.

    Returns:
        str: JSON-formatted search results.
    """
    try:
        # Load API key from credentials file
        credentials = load_yaml(CREDENTIALS_PATH)
        api_key = credentials.get('serp', {}).get('key')

        if use_google_shopping and not product_id:
            logger.warning("Google Shopping requires a product_id. Falling back to standard search.")
            use_google_shopping = False  # Fallback

        search_params = {
            "q": query,
            "api_key": api_key,
            "num": 10
        }

        if use_google_shopping:
            search_params["engine"] = "google_product"
            search_params["product_id"] = product_id

        search = GoogleSearch(search_params)
        results = search.get_dict()

        if "error" in results:
            return json.dumps({"error": results["error"]})

        formatted_results = format_product_results(results)
        return json.dumps({"products": formatted_results}, indent=2)

    except Exception as e:
        logger.error(f"Error in product_search: {e}")
        return json.dumps({"error": str(e)})

def format_product_results(results: dict) -> list:
    """Formats product search results from SerpAPI."""
    product_results = results.get("shopping_results", results.get("organic_results", []))

    formatted_results = []
    for result in product_results:
        try:
            formatted = {
                "title": result.get("title"),
                "link": result.get("link"),
                "price": result.get("price"),
                "store": result.get("source"),
            }
            formatted_results.append(formatted)
        except Exception as e:
            logger.error(f"Error formatting product result: {e}, Result: {result}")

    return formatted_results

if __name__ == "__main__":
    test_queries = [
        ("Dior foundation", None),
        ("iPhone 15", "123456789"),  # Example product ID
        ("Nike running shoes", None)
    ]
    for query, product_id in test_queries:
        results = product_search(query, product_id)
        print(f"Results for '{query}':\n{results}\n")
