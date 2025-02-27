# /Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/src/tools/manager.py

from src.config.log_config import logger
from pydantic import BaseModel, Field
import json
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Dict, Union, Any, List, Protocol, Tuple, Optional
from src.react.constants import Name
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import InputOutputTextPair
from src.config.setup import Config
from src.llm.gemini import generate

Observation = Union[str, Exception]

class Choice(BaseModel):
    name: str = Field(..., description="Name of the tool chosen.")
    reason: str = Field(..., description="Reason for choosing the tool.")
    input: str = Field("", description="Input for the chosen tool.")

class ToolFunction(Protocol):  # Protocol for Tool function type
    def __call__(self, query: str) -> Union[str, Tuple[int, str]]: ...

class Tool:
    def __init__(self, name: Name, description: str, func: ToolFunction) -> None:
        self.name = name
        self.description = description
        self.func = func

    def use(self, query: str) -> Union[str, Tuple[int, str]]:
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error using tool {self.name}: {e}")
            return e

class ResearchPhase(Enum):
    INDUSTRY = auto()
    USE_CASE = auto()
    RESOURCE = auto()

@dataclass
class ResearchContext:
    phase: ResearchPhase
    collected_data: dict
    current_focus: str

class Manager:
    def __init__(self, llm: GenerativeModel, model: GenerativeModel = None):
        self.llm = llm
        self.model = model
        self.tools: Dict[Name, Tool] = {}

    def ask_llm(self, prompt: str) -> str:
        """Interacts with the LLM."""
        try:
            logger.debug(f"Sending prompt to LLM: {prompt[:100]}...")
            # For GenerativeModel
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                logger.debug(f"LLM response received: {response.text[:100]}...")
                return response.text
            else:
                logger.error("LLM generation failed: empty response")
                raise ValueError("LLM generation failed: empty response")
                
        except Exception as e:
            logger.error(f"Error interacting with LLM: {e}")
            raise

    def act(self, choice: Choice) -> Observation:
        # Convert string name to enum if needed
        if isinstance(choice.name, str):
            try:
                tool_name = Name[choice.name.upper()]
            except (KeyError, AttributeError):
                raise ValueError(f"Tool {choice.name} not registered.")
        else:
            tool_name = choice.name
            
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered.")

        tool_result = self.tools[tool_name].use(choice.input)
        return tool_result

    def choose(self, query: str, available_tools: List[Tool], context: str = "") -> Choice:
        """Choose the best tool for the given query and context."""
        # Create a mapping of string names to enum values
        name_mapping = {}
        for tool in available_tools:
            # Map both the name attribute and the string representation of the enum
            name_mapping[tool.name.name.lower()] = tool.name  # e.g., "google_search" -> Name.GOOGLE_SEARCH
            name_mapping[str(tool.name).lower()] = tool.name  # e.g., "name.google_search" -> Name.GOOGLE_SEARCH
            # Also map simplified names that the LLM might return
            simple_name = tool.name.name.lower().replace('_', '')
            name_mapping[simple_name] = tool.name  # e.g., "googlesearch" -> Name.GOOGLE_SEARCH
        
        # Add NONE mapping
        name_mapping["none"] = Name.NONE
        
        # Format available tools for the prompt
        tools_string = "\n".join([
            f"- {tool.name.name.lower()}: {tool.description}" 
            for tool in available_tools
        ])
        
        prompt = f"""Choose the best tool for this task.
        Query: {query}
        Context: {context}
        
        Available Tools:
        {tools_string}
        
        You must respond with valid JSON in this exact format:
        {{
            "thought": "your detailed reasoning here",
            "tool": {{
                "name": "one_of_the_available_tool_names_exactly_as_shown_above",
                "input": "specific query for the tool"
            }}
        }}
        """

        try:
            response = self.ask_llm(prompt)
            
            # Debug the raw response
            logger.debug(f"Raw LLM response: {response}")
            
            parsed = json.loads(response)
            
            tool_choice = parsed.get("tool", {})
            raw_name = tool_choice.get("name", "none").lower()
            
            logger.debug(f"Raw tool name from LLM: {raw_name}")
            logger.debug(f"Available mappings: {list(name_mapping.keys())}")
            
            # Look for the tool name in our mapping
            tool_name = name_mapping.get(raw_name)
            
            # If not found, try to find a close match
            if not tool_name:
                for key in name_mapping:
                    if raw_name in key or key in raw_name:
                        tool_name = name_mapping[key]
                        logger.info(f"Found approximate match: {raw_name} -> {key} -> {tool_name}")
                        break
            
            # Default to NONE if still not found
            if not tool_name:
                logger.warning(f"Could not map '{raw_name}' to any registered tool. Using NONE.")
                tool_name = Name.NONE
            
            return Choice(
                name=tool_name.name if isinstance(tool_name, Name) else tool_name,
                reason=parsed.get("thought", "No reasoning provided"),
                input=tool_choice.get("input", query)
            )
            
        except Exception as e:
            logger.error(f"Error in tool selection: {e}")
            return Choice(
                name=Name.NONE.name,
                reason=f"Error in selection: {str(e)}",
                input=""
            )

    def advance_phase(self) -> None:
        """Advance to next research phase based on collected data."""
        phase_map = {
            ResearchPhase.INDUSTRY: ResearchPhase.USE_CASE,
            ResearchPhase.USE_CASE: ResearchPhase.RESOURCE,
            ResearchPhase.RESOURCE: ResearchPhase.INDUSTRY
        }
        self.research_context.phase = phase_map[self.research_context.phase]

    def force_tool_use(self, available_tools: List[Tool], context: str) -> Choice:  # Fixed type hints
        """Forces the use of a random tool when no tool is explicitly chosen by the LLM."""
        import random

        chosen_tool = random.choice(available_tools)
        thought = f"I am forcing use of {chosen_tool.name} to gather more information."
        input_query = f"Using {chosen_tool.name}, related to {context}"
        return Choice(name=chosen_tool.name, reason=thought, input=input_query)
    
    def register(self, name: Name, description: str, func: ToolFunction) -> None:
        """
        Register a tool with the given name, description, and function.
        
        Args:
            name (Name): The name of the tool.
            description (str): The description of the tool.
            func (ToolFunction): The function of the tool.
        """
        self.tools[name] = Tool(name, description, func)

def run() -> None:
    """Demonstrates Manager and Tool usage with comprehensive tests."""

    config = Config()
    gemini_model = GenerativeModel.from_pretrained(config.MODEL_NAME)

    manager = Manager()

    def mock_google_search(query: str) -> str:
         return json.dumps({"search_results": f"Results for {query}"})

    def mock_industry_report(query: str) -> str:
        return json.dumps({"report": f"Report on {query}"})

    def mock_competitor_analysis(query: str) -> str:
        return json.dumps({"competitors": ["Competitor A", "Competitor B"]})
    def mock_dataset_search(query: str) -> str:
        return json.dumps({"datasets": ["Dataset A", "Dataset B"]})

    manager.register(Name.GOOGLE, "Simple Google Search", mock_google_search)
    manager.register(Name.INDUSTRY_REPORT, "Finds industry reports", mock_industry_report)
    manager.register(Name.COMPETITOR_ANALYSIS, "Analyzes competitors", mock_competitor_analysis)
    manager.register(Name.DATASET_SEARCH, "Searches for datasets", mock_dataset_search)

    test_cases = [
        ("Market research for AI in cosmetics", [Name.GOOGLE, Name.INDUSTRY_REPORT], "Focus on Sephora"),
        ("Find competitors of L'Oreal", [Name.COMPETITOR_ANALYSIS, Name.GOOGLE], "Cosmetics industry"),
        ("Get datasets for customer segmentation", [Name.DATASET_SEARCH], "Luxury retail"),
        ("Invalid tool test", [Name.GOOGLE], ""),
        ("Error handling test", [Name.INDUSTRY_REPORT], "")
    ]

    for query, available_tools, context in test_cases:

        logger.info(f"Test Case: {query}")

        try:
            choice = manager.choose(query, available_tools, context)  
            if choice.name != Name.NONE:

                result = manager.act(choice)
                logger.info(f"Tool: {choice.name},  Result: {result}")

            else:
               logger.info("No tool chosen by LLM.") 


        except ValueError as e:
            logger.error(f"Error: {e}")

    try:
        invalid_choice = Choice(name="INVALID_TOOL", reason="Testing", input="test")  # Create an invalid choice
        manager.act(invalid_choice)

    except ValueError as e:
        logger.info(f"Successfully caught invalid tool error: {e}")

    def mock_tool_with_error(query):
        return (500, "Internal Server Error")

    manager.register(Name.ERROR_TOOL, "Simulates a tool error", mock_tool_with_error)

    try:

        error_choice = Choice(name=Name.ERROR_TOOL, reason="Test error handling.", input="Error")
        result = manager.act(error_choice)
        logger.info(f"Tool with Error Result: {result}")

    except Exception as e: 
         logger.error(f"Unexpected error: {e}")


    for query, tools, context in test_cases:
        try:
            choice = manager.choose(query, tools, context)
            result = manager.act(choice)
            logger.info(f"Query: {query}, Tool: {choice.name}, Result: {result}")  

        except ValueError as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    run()