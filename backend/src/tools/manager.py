from src.config.log_config import logger
from pydantic import BaseModel, Field
import json
import re
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
        self.function = func

    def __str__(self) -> str:
        """String representation of the tool."""
        return f"{self.name}: {self.description}"

    def use(self, query: str) -> Union[str, Tuple[int, str]]:
        try:
            logger.info(f"Using tool {self.name} with query: {query[:100]}...")
            return self.function(query)
        except Exception as e:
            logger.error(f"Error using tool {self.name}: {e}")
            return f"Error in {self.name}: {str(e)}"

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
        self.model = model or llm
        self.tools: Dict[Name, Tool] = {}
        
        # Add special mapping for common name mismatches
        self.name_mappings = {
            # Common variations that the LLM might return
            "google": Name.GOOGLE_SEARCH,
            "googlesearch": Name.GOOGLE_SEARCH,
            "search": Name.GOOGLE_SEARCH,
            "websearch": Name.GOOGLE_SEARCH,
            
            "competitor": Name.COMPETITOR_ANALYSIS,
            "competitors": Name.COMPETITOR_ANALYSIS,
            "competitoranalysis": Name.COMPETITOR_ANALYSIS,
            "competition": Name.COMPETITOR_ANALYSIS,
            
            "industry": Name.INDUSTRY_REPORT,
            "industryreport": Name.INDUSTRY_REPORT,
            "report": Name.INDUSTRY_REPORT,
            "reports": Name.INDUSTRY_REPORT,
            
            "product": Name.PRODUCT_SEARCH,
            "products": Name.PRODUCT_SEARCH,
            "productsearch": Name.PRODUCT_SEARCH,
            
            "dataset": Name.DATASET_SEARCH,
            "datasetsearch": Name.DATASET_SEARCH,
            "datasets": Name.DATASET_SEARCH,
            "data": Name.DATASET_SEARCH,
            
            "trends": Name.GOOGLE_TRENDS,
            "googletrends": Name.GOOGLE_TRENDS,
            
            "brainstorm": Name.BRAINSTORM_USE_CASES,
            "brainstormcases": Name.BRAINSTORM_USE_CASES,
            "usecases": Name.BRAINSTORM_USE_CASES,
            
            "none": Name.NONE,
            "final": Name.NONE,
            "finalanswer": Name.NONE
        }

    def ask_llm(self, prompt: str) -> str:
        """Interacts with the LLM."""
        try:
            logger.debug(f"Sending prompt to LLM: {prompt[:100]}...")
            # For GenerativeModel
            response = self.model.generate_content(prompt)
            
            if response and hasattr(response, 'text') and response.text:
                logger.debug(f"LLM response received: {response.text[:100]}...")
                return response.text
            else:
                logger.error("LLM generation failed: empty response")
                raise ValueError("LLM generation failed: empty response")
                
        except Exception as e:
            logger.error(f"Error interacting with LLM: {e}")
            raise

    def act(self, choice: Choice) -> Observation:
        """Execute the chosen tool."""
        # Convert string name to enum if needed
        tool_name = None
        
        if isinstance(choice.name, str):
            try:
                # First try direct mapping from our custom dictionary
                norm_name = choice.name.lower().replace('_', '')
                if norm_name in self.name_mappings:
                    tool_name = self.name_mappings[norm_name]
                else:
                    # Try to find by exact match in enum
                    try:
                        tool_name = Name[choice.name.upper()]
                    except KeyError:
                        # Try to find close matches among available tools
                        for name_enum in Name:
                            if choice.name.lower() in name_enum.name.lower():
                                tool_name = name_enum
                                logger.info(f"Found approximate match: {choice.name} -> {name_enum}")
                                break
            except (KeyError, AttributeError):
                raise ValueError(f"Tool {choice.name} not registered or recognized.")
        else:
            tool_name = choice.name
        
        if not tool_name:
            raise ValueError(f"Could not map tool name '{choice.name}' to any registered tool.")
            
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered in available tools.")

        tool_result = self.tools[tool_name].use(choice.input)
        return tool_result

    def choose(self, query: str, available_tools: List[Tool], context: str = "") -> Choice:
        """Choose the best tool for the given query and context."""
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
            
            # Try to extract JSON even if it's not perfectly formatted
            json_match = re.search(r'({[\s\S]*})', response)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up common JSON formatting errors
                    json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Fix unquoted keys
                    json_str = re.sub(r',\s*}', r'}', json_str)  # Fix trailing commas
                    try:
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError:
                        raise ValueError(f"Could not parse LLM response as JSON: {response}")
            else:
                raise ValueError(f"No JSON found in LLM response: {response}")
            
            tool_choice = parsed.get("tool", {})
            raw_name = tool_choice.get("name", "none").lower().replace('_', '')
            
            logger.debug(f"Raw tool name from LLM: {raw_name}")
            
            # Try our custom mapping first
            if raw_name in self.name_mappings:
                tool_name = self.name_mappings[raw_name]
                logger.info(f"Mapped '{raw_name}' to {tool_name} using custom mapping")
            else:
                # Try to find a close match among available tools
                for tool in available_tools:
                    tool_str_name = tool.name.name.lower()
                    if raw_name in tool_str_name or tool_str_name in raw_name:
                        tool_name = tool.name
                        logger.info(f"Mapped '{raw_name}' to {tool_name} by string similarity")
                        break
                else:
                    # If no match found, default to first available tool or NONE
                    tool_name = available_tools[0].name if available_tools else Name.NONE
                    logger.warning(f"No matching tool found for '{raw_name}'. Defaulting to {tool_name}")
            
            # Ensure we have a valid input query
            tool_input = tool_choice.get("input", "").strip()
            if not tool_input:
                tool_input = query  # Fall back to original query if empty
            
            return Choice(
                name=tool_name.name if isinstance(tool_name, Name) else str(tool_name),
                reason=parsed.get("thought", "No reasoning provided"),
                input=tool_input
            )
            
        except Exception as e:
            logger.error(f"Error in tool selection: {e}")
            # Default to the first available tool as fallback
            default_tool = available_tools[0] if available_tools else None
            if default_tool:
                return Choice(
                    name=default_tool.name.name,
                    reason=f"Default choice due to error: {str(e)}",
                    input=query
                )
            else:
                return Choice(
                    name=Name.NONE.name,
                    reason=f"Error in selection and no default available: {str(e)}",
                    input=""
                )

    def get_tool_by_name(self, name: Union[str, Name]) -> Optional[Tool]:
        """Get a tool by its name."""
        if isinstance(name, str):
            # Try to find the tool by string name
            for tool_name, tool in self.tools.items():
                if tool_name.name.lower() == name.lower():
                    return tool
                
            # Try name mappings
            norm_name = name.lower().replace('_', '')
            if norm_name in self.name_mappings:
                mapped_name = self.name_mappings[norm_name]
                return self.tools.get(mapped_name)
        else:
            # Direct lookup by enum
            return self.tools.get(name)
        
        return None

    def force_tool_selection(self, tool_name: Union[str, Name], query: str) -> Choice:
        """Directly select a specific tool without LLM reasoning."""
        # Locate the tool by name
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            logger.error(f"Forced tool selection failed: {tool_name} not found")
            raise ValueError(f"Tool {tool_name} not found for forced selection")
            
        return Choice(
            name=tool.name.name,
            reason=f"Tool selected by predefined sequence: {tool.name}",
            input=query
        )
    
    def register(self, name: Name, description: str, function: ToolFunction) -> None:
        """
        Register a tool with the given name, description, and function.
        
        Args:
            name (Name): The name of the tool.
            description (str): The description of the tool.
            function (ToolFunction): The function of the tool.
        """
        self.tools[name] = Tool(name, description, function)

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