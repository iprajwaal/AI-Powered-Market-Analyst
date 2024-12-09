from src.tools import *  # Import all your tool functions
from src.config.logging import logger
from pydantic import BaseModel, Field
import json
from typing import Callable, Dict, Union, Any, List, Protocol, Tuple
# from src.utils.constants import Name # Import Name enum
from vertexai.preview.language_models import TextGenerationModel  # For Gemini
from src.config.setup import Config

Observation = Union[str, Exception]
class Name(Enum):
    GOOGLE = "google"
    INDUSTRY_REPORT = "industry_report"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    DATASET_SEARCH = "dataset_search"
    ERROR_TOOL = "error_tool"
    NONE = "none"
# Name Enum (should be defined in a common place, e.g., utils or constants)

class Choice(BaseModel):
    """ 
    Available tool choices. 
    """
    name: Name = Field(...)
    reason: str = Field(...)
    input: str = Field(...)  # Tool input

class ToolFunction(Protocol):  # Protocol for Tool function type
    def __call__(self, query: str) -> Union[str, Tuple[int, str]]: ...



class Tool:
    """
    Represents a tool with a name, description, and function to exectute.
    """
    def __init__(self, name: Name, description: str, func: ToolFunction) -> None:
        self.name = name
        self.description = description
        self.func = func

    def use(self, query: str) -> Union[str, Tuple[int, str]]:
        """
        Executes the tool function with the given query.
        """
        try:
            return self.func(query)
        except Exception as e:
            logger.error(f"Error using tool {self.name}: {e}")
            return e


class Manager:
    """
    Manages the tools and their execution.
    """
    def __init__(self, tools: Dict[Name, Tool] = None) -> None:
        self.tools = tools or {}
        self.model = TextGenerationModel(Config().MODEL_NAME) 

    def ask_llm(self, prompt: str) -> str:
        """
        Mock function to simulate interaction with an LLM.
        Replace this with the actual implementation.
        """
        response = self.model.generate(prompt)
        return response

    def register(self, name: Name, description: str, func: ToolFunction) -> None:
        """ 
        Registers a new tool with the manager.
        """
        self.tools[name] = Tool(name, description, func)

    def act(self, choice: Choice) -> Observation: 
        """
        Executes the selected tool to act on the query.

        Parameters:
            name (Name): The name of the tool to use.
            query (str): The query to act on.

        Returns:
            Observation: The result of the tool execution
        """
        if choice.name not in self.tools:
            raise ValueError(f"Tool {choice.name} not registered.")

        tool_result = self.tools[choice.name].use(choice.input)
        try:
            parsed_result = json.loads(tool_result)
            logged_result = json.dumps(parsed_result, indent=2)
        except (json.JSONDecodeError, TypeError): # Handle non-JSON or errors.
             logged_result = tool_result

        logger.info(f"Tool {choice.name} returned: {logged_result}") #Detailed result logs.
        return tool_result


    def choose(self, query: str, available_tools: List[Name], context: str = "") -> Choice:
        """Chooses the best tool using the LLM."""

        tools_string = ", ".join(tool.name.lower().replace("_", " ") for tool in available_tools)

        prompt = f"""You are a helpful AI assistant. Choose the best tool to answer the following query.
        Query: {query}
        Context: {context} 
        Available Tools: {tools_string}
        Tool Descriptions: {[str(tool) for tool in self.tools.values() if tool.name in available_tools]}


        Respond in this JSON format ONLY:
        {{
        llm_response = self.ask_llm(prompt)  # Replace with your actual LLM interaction code
            "tool": {{ # Use 'tool' key here
                "name": "tool_name",
                "input": "Specific tool input"
            }}
        }}

        Examples:
        {{"thought": "I need to find general information about the company.", "tool": {{"name": "google_search", "input": "Sephora company overview"}}}}
        {{"thought": "I need to find industry reports.", "tool": {{"name": "industry_report", "input": "Cosmetics industry market analysis"}}}}
        {{"thought": "Time to brainstorm use cases now that I have market data.", "tool": {{"name": "none", "input": ""}}}}

        """

        llm_response = self.ask_llm(prompt)  # Replace with your actual LLM interaction code
        try:
            parsed_response = json.loads(llm_response)
            tool_choice = parsed_response["tool"]  # Access the "tool" dictionary

            if tool_choice["name"] == "none" or Name[tool_choice["name"].upper()] in available_tools:

                return Choice(name=Name[tool_choice["name"].upper()] if tool_choice["name"] != "none" else Name.NONE, 
                            reason=parsed_response["thought"], input=tool_choice["input"])
            else:
                logger.warning(f"Invalid tool choice from LLM: {tool_choice['name']}")
                raise ValueError("Invalid tool name or tool not available.") #Handle appropriately.

        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Invalid LLM response: {llm_response}, error: {e}")
            raise ValueError("Could not parse LLM response for tool choice.")
        
    def force_tool_use(self, available_tools, context):
        import random

        chosen_tool = random.choice(available_tools)
        thought = f"I am forcing use of {chosen_tool} to gather more information."
        input_query = f"Using {chosen_tool}, related to {context}" # Or other suitable input.  Could be original query.
        return Choice(name = chosen_tool, reason=thought, input= input_query)


def run() -> None:
    """Demonstrates Manager and Tool usage with comprehensive tests."""

    config = Config()
    gemini_model = TextGenerationModel(config.MODEL_NAME)

    manager = Manager(model = gemini_model)

    def mock_google_search(query: str) -> str:
         return json.dumps({"search_results": f"Results for {query}"})

    def mock_industry_report(query: str) -> str:
        return json.dumps({"report": f"Report on {query}"})

    def mock_competitor_analysis(query: str) -> str:
        return json.dumps({"competitors": ["Competitor A", "Competitor B"]})


    # ... define other mock tool functions.

    manager.register(Name.GOOGLE, "Simple Google Search", mock_google_search)
    manager.register(Name.INDUSTRY_REPORT, "Finds industry reports", mock_industry_report)
    manager.register(Name.COMPETITOR_ANALYSIS, "Analyzes competitors", mock_competitor_analysis)

    # ... register other mock tools

    test_cases = [
        ("Market research for AI in cosmetics", [Name.GOOGLE, Name.INDUSTRY_REPORT], "Focus on Sephora"),
        ("Find competitors of L'Oreal", [Name.COMPETITOR_ANALYSIS, Name.GOOGLE], "Cosmetics industry"),
        ("Get datasets for customer segmentation", [Name.DATASET_SEARCH], "Luxury retail"),
        # ... More test cases with different queries, tool combinations, and contexts.
        ("Invalid tool test", [Name.GOOGLE], ""), #Test invalid tool choice.
        ("Error handling test", [Name.INDUSTRY_REPORT], "") #Simulate tool error
    ]

    for query, available_tools, context in test_cases:

        logger.info(f"Test Case: {query}") # Separate test case logs.

        try:
            choice = manager.choose(query, available_tools, context)  # Use available_tools
            if choice.name != Name.NONE: # Only call act() if a tool is chosen.

                result = manager.act(choice)
                logger.info(f"Tool: {choice.name},  Result: {result}")

            else:
               logger.info("No tool chosen by LLM.") # Log this explicitly


        except ValueError as e:
            logger.error(f"Error: {e}")

    # Test handling invalid tool choices and tool execution errors.
    # These are especially important to ensure your error handling works correctly.

    try:
        invalid_choice = Choice(name="INVALID_TOOL", reason="Testing", input="test")  # Create an invalid choice
        manager.act(invalid_choice) # Should raise exception

    except ValueError as e:
        logger.info(f"Successfully caught invalid tool error: {e}") # Expected behavior.


    # Simulate a tool returning a tuple indicating error
    def mock_tool_with_error(query):
        return (500, "Internal Server Error") # Example error tuple

    manager.register(Name.ERROR_TOOL, "Simulates a tool error", mock_tool_with_error)

    try:

        error_choice = Choice(name=Name.ERROR_TOOL, reason="Test error handling.", input="Error")
        result = manager.act(error_choice)
        logger.info(f"Tool with Error Result: {result}") # Check the error message.

    except Exception as e: #Shouldn't reach here
         logger.error(f"Unexpected error: {e}")


    test_cases = [
        ("Market research for AI in cosmetics", [Name.GOOGLE, Name.INDUSTRY_REPORT], "Focus on Sephora"),
        # ... more test cases with different queries, available tools, and context
    ]

    for query, tools, context in test_cases:
        try:
            choice = manager.choose(query, tools, context)
            result = manager.act(choice)
            logger.info(f"Query: {query}, Tool: {choice.name}, Result: {result}")  # Log results

        except ValueError as e:
            logger.error(f"Error: {e}")


if __name__ == "__main__":
    run()