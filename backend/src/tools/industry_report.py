from src.tools.google_search import google_search
from typing import Dict, List, Any, Tuple, Union
import logging
from src.config.setup import Config
import requests
import PyPDF2
import json
import os

config = Config()
logger = logging.getLogger(__name__) 

def industry_report_search(query: str) -> str:
    """Searches for industry reports and extracts text from PDFs."""
    try:
        search_query = f"{query} industry report filetype:pdf"
        initial_search_json = google_search(search_query)
        try:
            initial_search = json.loads(initial_search_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Could not parse initial search results: {e}"})


        if "error" in initial_search:
            return json.dumps({"error": initial_search["error"]})

        report_links = initial_search.get("organic_results", [])

        extracted_text = ""
        for link in report_links:
            if ".pdf" in link.get("link", ""):
                try:
                    pdf_text = extract_text_from_pdf(link["link"])

                    if isinstance(pdf_text, str):
                        extracted_text += pdf_text
                    elif isinstance(pdf_text, tuple):
                        status_code, error_message = pdf_text
                        logger.error(f"PDF extraction error: {error_message}, Link: {link['link']}") 
    
                        extracted_text += f"Error extracting from {link['link']}: {error_message}\n"
                    else: 
                        logger.error(f"Unexpected return type from extract_text_from_pdf: {type(pdf_text)}")
                        extracted_text += f"Unexpected error extracting from {link['link']}\n"


                except Exception as e:  
                    logger.error(f"Error extracting PDF text: {e}, Link: {link.get('link', 'N/A')}")
                    extracted_text += f"Error extracting from {link['link']}: {e}\n"  


        return json.dumps({"extracted_text": extracted_text, "report_links": report_links}, indent=2)

    except Exception as e: 
        logger.error(f"Error in industry_report_search: {e}")
        return json.dumps({"error": str(e)})  


def extract_text_from_pdf(pdf_link: str) -> Union[str, Tuple[int, str]]: 
    """
    Extracts text from a PDF link, handling various error conditions.
    """

    try:
        response = requests.get(pdf_link, stream=True, timeout=10) 
        response.raise_for_status()  

        with open("temp.pdf", "wb") as temp_pdf:
            temp_pdf.write(response.content)

        reader = PyPDF2.PdfReader("temp.pdf")

        all_text = ""
        for page_num in range(len(reader.pages)):
             page = reader.pages[page_num]
             parts = []


             def visitor_body(text, cm, tm, fontDict, fontSize):
                  y = tm[5] 
                  if y > 50 and y < 720: 
                     parts.append(text)

             page.extract_text(visitor_text=visitor_body) 
             page_text = "".join(parts)
             all_text += page_text 

        os.remove("temp.pdf")  

        return all_text

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        return f"Error downloading PDF: {e}"
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"Error reading PDF: {e}")  
        return f"Error reading PDF: {e}"
    except Exception as e: 
        logger.exception(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"


if __name__ == "__main__":
    test_queries = [
        "Cosmetics industry",
        "Luxury handbag market",
        "AI in finance"  # Example
    ]
    for query in test_queries:
        results = industry_report_search(query)
        print(f"Industry Report Results for '{query}':\n{results}\n")