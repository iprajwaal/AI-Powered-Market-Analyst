from src.tools import * 
from src.config.log_config import logger
from pydantic import BaseModel, Field
import json
from enum import Enum, auto
from dataclasses import dataclass
from typing import Callable, Dict, Union, Any, List, Protocol, Tuple, Optional
from src.utils.constants import Name
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import InputOutputTextPair
from src.config.setup import Config
from src.llm.gemini import generate

Observation = Union[str, Exception]


class Name(Enum):
    GOOGLE_SEARCH = "google_search"
    INDUSTRY_REPORT = "industry_report"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    DATASET_SEARCH = "dataset_search"
    BRAINSTORM_USE_CASES = "brainstorm_use_cases"
    PRODUCT_SEARCH = "product_search"
    GOOGLE_TRENDS = "google_trends"
    NONE = "none"


class Choice(BaseModel):
    name: Name
    reason: str
    input: str = ""


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
    def __init__(self, llm: TextGenerationModel, model: TextGenerationModel = None):
        self.llm = llm
        self.model = model
        self.tools: Dict[Name, Tool] = {}

    def ask_llm(self, prompt: str) -> str:
        """Interacts with the LLM."""
        try:
            response = generate(model=self.model, prompt=prompt)

            if response is None:
                logger.error("LLM generation failed.")
                raise ValueError("LLM generation failed.")
            
            return response
        except Exception as e:
            logger.error(f"Error interacting with LLM: {e}")
            raise

    def register(self, name: Name, description: str, func: ToolFunction) -> None:
        self.tools[name] = Tool(name, description, func)

    def act(self, choice: Choice) -> Observation:
        if choice.name not in self.tools:
            raise ValueError(f"Tool {choice.name} not registered.")

        tool_result = self.tools[choice.name].use(choice.input)

        try:
            parsed_result = json.loads(tool_result)
            logged_result = json.dumps(parsed_result, indent=2)
        except (json.JSONDecodeError, TypeError):
            logged_result = tool_result

        logger.info(f"Tool {choice.name} returned: {logged_result}")
        return tool_result

    def choose(self, query: str, available_tools: List[Tool], context: str = "") -> Choice:
        tools_string = ", ".join([
            f"{tool.name.value}: {tool.description}" 
            for tool in available_tools
        ])
        
        prompt = f"""Choose the best tool for this task.
        Query: {query}
        Context: {context}
        Available Tools: {tools_string}

        Respond in JSON format:
        {{
            "thought": "reasoning",
            "tool": {{
                "name": "TOOL_NAME",
                "input": "specific query"
            }}
        }}"""

        try:
            response = self.ask_llm(prompt)
            parsed = json.loads(response)
            
            tool_choice = parsed.get("tool", {})
            try:
                tool_name = Name[tool_choice.get("name", "NONE")]
            except KeyError:
                tool_name = Name.NONE
                
            return Choice(
                name=tool_name,
                reason=parsed.get("thought", "No reasoning provided"),
                input=tool_choice.get("input", query)
            )
            
        except Exception as e:
            logger.error(f"Error in tool selection: {e}")
            return Choice(
                name=Name.NONE,
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


def run() -> None:
    """Demonstrates Manager and Tool usage with comprehensive tests."""

    config = Config()
    gemini_model = TextGenerationModel.from_pretrained(config.MODEL_NAME)

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