from src.tools.google_search import google_search  
from typing import Dict, List, Any
import json
import logging
from serpapi import GoogleSearch
from src.config import config

logger = logging.getLogger(__name__)

def competitor_analysis(query: str) -> str:
    """
    Performs competitor analysis using Google Search and Google Trends.
    """
    try:
        competitor_search_query = f"{query} competitors"
        competitors_search_json = google_search(competitor_search_query)
        competitors_search = json.loads(competitors_search_json)


        if "error" in competitors_search:
            competitors = []
        else:
            competitors = extract_competitors_from_search(competitors_search)

        trends_search_query = query
        trends_results_json = google_trends_search(trends_search_query)
        trends_results = json.loads(trends_results_json)


        if "error" in trends_results:
            rising_queries = []
        else:
             rising_queries = trends_results.get("rising_queries", [])

        return json.dumps({
            "direct_competitors": competitors, 
            "rising_queries": rising_queries
        }, indent=2)

    except Exception as e:
        logger.error(f"Error in competitor_analysis: {e}")
        return json.dumps({"error": str(e)})


def extract_competitors_from_search(search_results: Dict) -> List[str]:
    """
    Extract direct competitor names from the Google Search results.  Improve this using NLP techniques or better prompt engineering in the future.
    """

    competitors = []
    organic_results = search_results.get('organic_results', [])

    for result in organic_results:
        snippet = result.get('snippet', "")
        keywords = ["competitors", "rivals", "alternatives"]
        if any(keyword in snippet.lower() for keyword in keywords):
            competitors.append(snippet)
    return competitors


def google_trends_search(query: str) -> str:
    """Performs a Google Trends search using SerpAPI."""
    try:
        search = GoogleSearch({"engine": "google_trends", "q": query, "api_key": config.SERPAPI_KEY})
        results = search.get_json()
        return json.dumps(results, indent=2)
    except Exception as e:  #Handle errors
        logger.error(f"Error in google_trends_search: {e}")
        return json.dumps({"error": str(e)})
    

if __name__ == "__main__":
    test_queries = [
        "Sephora",
        "L'Oreal",
        "Chanel"  # Example
    ]
    for query in test_queries:
        results = competitor_analysis(query)
        print(f"Competitor Analysis Results for '{query}':\n{results}\n")