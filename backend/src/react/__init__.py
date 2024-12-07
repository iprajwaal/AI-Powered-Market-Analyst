import os
import json
from enum import Enum, auto
from src.utils.io import read_file
from src.config.setup import Config
from src.llm.gemini import generate
from src.config.logging import logger 
from pydantic import BaseModel, Field
from typing import Dict, List, Callable, Protocol
from vertexai.generative_models import GeneralionModel, Part
from src.tools.serp import search as google_search
# ... to be imported other tools (industry_report, competitor_analysis, dataset_search, brainstorm_use_cases, product_search, google_trends)

Observation = Union[str, Exception]

PROMPT_TEMPLATE_PATH = [
    "./data/react.txt",
    "./backend/data/react.txt"
]

class Name(Enum):
    GOOGLE = auto()
    INDUSTRY_REPORT = auto()
    COMPETITOR_ANALYSIS = auto()
    DATASET_SEARCH = auto()
    BRAINSTORM_USE_CASES = auto()
    PRODUCT_SEARCH = auto()  
    GOOGLE_TRENDS = auto()
    NONE = auto()

    def __str__(self) -> str:
        """
        representation of the tool name.
        """
        return self.name.lower().replace("_", " ")
    
class Choice(BaseModel):
    """
    Choice of tools with a resone for the choice.
    """
    name: Name = Field(..., description="Name of the tool chosen.")
    reason: str = Field(..., description="Reason for choosing the tool.")

class Message(BaseModel):
    """
    Represents a message with sender role and content.
    """
    role: str = Field(..., description="Role of the sender.")
    content: str = Field(..., description="Content of the message.")

logger.basicConfig(level=logger.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ToolFunction(Protocol):  # Protocol for Tool function type
    def __call__(self, query: str) -> Observation: ...

class Tool:
    """
    Tool class to represent a tool with its name, description, and function.
    """
    def __init__(self, name: Name, description: str, func: Callable[[str], str]):
        self.name = name
        self.description = description
        self.function = function

    def __str__(self) -> str:
        """
        representation of the tool.
        """
        return f"{self.name}: {self.description}"

    def use(self, query: str) -> Observation:
        """Executes the tool's function with the provided query."""

        log = logger.getLogger(__name__)
        try:
            log.info(f"Using tool {self.name} with query: {query}")
            result = self.func(query)  # Use self.func consistently
            log.debug(f"Tool {self.name} returned: {result}")
            return result

        except Exception as e:  # Or handle more specific exceptions
            log.exception(f"Error executing tool {self.name}: {e}") # Log the exception
            return f"Error in {self.name}: {e}" # Return an informative error message