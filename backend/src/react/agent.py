# /Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/src/react/agent.py

import os
import json
from enum import Enum, auto
from src.utils.io import read_file, write_to_file
from src.config.setup import Config
from src.llm.gemini import generate
from src.config.log_config import logger 
from pydantic import BaseModel, Field
from typing import Dict, List, Callable, Protocol, Union, Any, Optional
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview.generative_models import Part
from src.tools.google_search import google_search
from src.tools.industry_report import industry_report_search as industry_report_tool
from src.tools.competitor_analysis import competitor_analysis as competitor_analysis_tool
from src.tools.dataset_search import dataset_search as dataset_search_tool
from src.tools.product_search import product_search as search_google_products
from src.tools.competitor_analysis import google_trends_search as search_google_trends
from src.tools.manager import Manager
import logging
from src.react.constants import Name

Observation = Union[str, Exception]

PROMPT_TEMPLATE_PATH = [
    "./data/react.txt",
    "./backend/data/react.txt"
]
OUTPUT_TRACE_PATH = "/Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/data/output/trace.json"

    
class Choice(BaseModel):
    """
    Choice of tools with a reason for the choice.
    """
    name: str = Field(..., description="Name of the tool chosen.")
    reason: str = Field(..., description="Reason for choosing the tool.")

class Message(BaseModel):
    """
    Represents a message with sender role and content.
    """
    role: str = Field(..., description="Role of the sender.")
    content: str = Field(..., description="Content of the message.")

class ToolFunction(Protocol):  # Protocol for Tool function type
    def __call__(self, query: str) -> Observation: ...

class Tool:
    """
    Tool class to represent a tool with its name, description, and function.
    """
    def __init__(self, name: Name, description: str, func: ToolFunction):
        self.name = name
        self.description = description
        self.function = func

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
            result = self.function(query) 
            log.debug(f"Tool {self.name} returned: {result}")
            return result

        except Exception as e:  
            log.exception(f"Error executing tool {self.name}: {e}")
            return f"Error in {self.name}: {e}"

class Agent:
    def __init__(self, model: GenerativeModel, manager: Optional[Manager] = None) -> None:
        """
        Initialize the Agent object with the given model, tools, and messages.
        
        Args:
            model (GenerativeModel): The model to use for text generation.
        """
        self.model = model
        self.tools: Dict[Name, Tool] = {}
        self.manager = manager or Manager(llm=model, model=model)
        self.messages = []  # Initialize messages as a list
        self.tools: Dict[Name, Tool] = {}
        self.query = ""
        self.max_iterations = 15
        self.current_iteration = 0
        self.min_iterations = 5
        self.template = self._load_template()

    def _load_template(self) -> str:
        """
        Load the prompt template from the file.

        Returns:
            str: The prompt template.
        Raises:
            FileNotFoundError: If the template file is not found.
        """
        for path in PROMPT_TEMPLATE_PATH:
            if os.path.exists(path):
                logger.info(f"Found template file at {path}")
                return read_file(path)
            
        logger.error("No template file found in any default location.")
        raise FileNotFoundError("No template file found in any default location.")
    
    def register(self, name: Name, description: str, function: ToolFunction) -> None:
        """
        Register a tool with the given name, description, and function.

        Args:
            name (Name): The name of the tool.
            description (str): The description of the tool.
            function (ToolFunction): The function of the tool.
        """
        tool = Tool(name, description, function)
        self.tools[name] = tool
        self.manager.tools[name] = tool

    def trace(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role (str): The role of the sender.
            content (str): The content of the message.
        """
        if role != "system":
            self.messages.append(Message(role=role, content=content))
        write_to_file(path=OUTPUT_TRACE_PATH, content=f'{role}: {content}\n')

    def get_history(self) -> str:
        """
        Retrive the conversation history.

        Returns:
            str: formatted conversation history.
        """
        return "\n".join([f"{m.role}: {m.content}" for m in self.messages])
    
    def think(self) -> None:
        """Main reasoning loop."""

        self.current_iteration += 1
        logger.info(f"Thinking iteration {self.current_iteration}...")
        write_to_file(path = OUTPUT_TRACE_PATH, content = f"\n{'='*50}\nThinking iteration {self.current_iteration}\n{'='*50}\n")

        # Debug info
        logger.debug(f"Available tools: {[t.name for t in self.tools.values()]}")
        
        if self.current_iteration >= self.max_iterations:
            logger.info("Max iterations reached. Stopping!")
            self.generate_final_answer()
            return

        # Get available tools (excluding BRAINSTORM_USE_CASES)
        available_tools = [self.tools[tool_name] for tool_name in self.tools if tool_name != Name.BRAINSTORM_USE_CASES]
        
        # More debug info
        logger.debug(f"Filtered tools for selection: {[t.name for t in available_tools]}")

        try:
            logger.debug(f"Calling manager.choose with query: {self.query}")
            choice = self.manager.choose(self.query, available_tools, self.get_history())
            logger.debug(f"Manager selected tool: {choice.name} with reason: {choice.reason}")

            if choice.name == Name.NONE:
                if self.current_iteration >= self.min_iterations:
                    self.generate_final_answer()  
                    return 
                else:
                    self.trace("assistant", "Thought: I need more information before finalizing.")
            
                    if available_tools:
            
                        choice = self.manager.force_tool_use(available_tools, self.get_history())
                    else:
                        self.generate_final_answer() 
                        return

            elif choice.name == Name.BRAINSTORM_USE_CASES: 
                self.brainstorm(choice.input)

            else:  
                self.act(choice) 

        except ValueError as e:  
            logger.error(f"Error choosing or using tool: {e}")
            self.trace("system", f"Error: {e}")
            self.think()  # Try again on the next iteration


    def decide(self, response: str) -> None:
        """
        Decides the next action based on the response from the model.

        Args:
            response (str): The response from the model.
        """
        try:
            parsed_response = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {response}. Error: {str(e)}")
            self.trace("assistant", "I encountered an error in processing. Let me try again.")
            self.think()
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            self.trace("assistant", "I encountered an unexpected error. Let me try a different approach.")
            self.think()

        if "action" in parsed_response:
            action = parsed_response["action"]
            tool_name = Name[action["name"].upper()]

            if tool_name == Name.NONE:
                logger.info("No tool selected. Stopping!")
                self.generate_final_answer()
            elif tool_name == Name.BRAINSTORM_USE_CASES:
                self.trace("assistant", "Action: Brainstorming use cases")
                self.brainstorm(action["input"])
            else:  # Handle other tools
                self.trace("assistant", f"Action: Using {tool_name} tool")
                self.act(tool_name, action.get("input", self.query))

    def brainstorm(self, context: str) -> None:
        """Handeling the brainstorming action."""

        prompt = f"""Based on the following context, brainstorm innovative AI/GenAI use cases:

        Context: {context}

        Provide a list of use cases in JSON format, where each use case has a "use_case" description and an empty "resources" list (to be filled later).
        Example:
        ```json
        [
            {{"use_case": "Personalized product recommendations based on customer style preferences", "resources": []}},
            {{"use_case": "Virtual try-on experience using augmented reality", "resources": []}}
        ]
        ```
        """
        use_cases_json = self.ask_gemini(prompt)
        try:
            use_cases = json.loads(use_cases_json)
            self.trace("system", f"Brainstormed Use Cases: {json.dumps(use_cases, indent=2)}")
            self.think()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse use cases: {use_cases_json}. Error: {str(e)}")
            self.trace("assistant", "I encountered an error while brainstorming use cases.Let me try another approach.")
            self.think()

    def act(self, choice: Choice) -> None:
        """
        Executes the chosen tool and processes the result.
        
        Args:
            choice (Choice): The choice of tool to act on.
            
        Returns:
            None
        """
        # Ensure we're working with a string name
        tool_name_str = choice.name
        
        # Convert to Name enum if possible (for better type safety)
        try:
            if isinstance(tool_name_str, str):
                tool_name = Name[tool_name_str.upper()]
            else:
                tool_name = tool_name_str
        except (KeyError, ValueError):
            logger.error(f"Invalid tool name: {tool_name_str}")
            self.trace("system", f"Error: Invalid tool name {tool_name_str}")
            self.think()
            return
        
        logger.debug(f"Looking up tool: {tool_name}, available tools: {list(self.tools.keys())}")
        tool = self.tools.get(tool_name)
        
        if not tool:
            logger.error(f"No tool registered for: {tool_name}")
            self.trace("system", f"Error: Tool {tool_name} not found")
            self.think()
            return

        try:
            query = choice.input or self.query  # Use provided input or fall back to original query
            result = tool.use(query)

            if isinstance(result, tuple):
                status_code, error_message = result
                observation = {"error": f"Tool {tool_name} failed with status {status_code}: {error_message}"}
                
            elif isinstance(result, dict):
                observation = result
            elif isinstance(result, str):
                try:
                    observation = json.loads(result)
                except json.JSONDecodeError:
                    observation = {"result": result}  # Store as plain text if not JSON

            else:
                observation = {"error": f"Unexpected tool output type: {type(result)}", "raw_output": str(result)}

            observation_message = f"Observation from {tool_name}: {json.dumps(observation, indent=2)}"
            self.trace("system", observation_message) 

            # Store the observation for the history
            self.messages.append(Message(role="system", content=json.dumps(observation)))

            # Continue thinking
            self.think()

        except Exception as e:
            error_message = f"Unexpected error using {tool_name}: {e}"
            logger.exception(error_message)
            self.trace("system", error_message)
            self.think()

    def generate_final_answer(self) -> None:
        """Generates the final structured answer."""

        industry_overview = ""
        competitor_analysis = ""
        potential_use_cases = []

        for message in self.messages:
            if message.role == "system": 
                try:
                    observation = json.loads(message.content)
                    if "report_links" in observation:
                        industry_overview += f"Industry Report Links: {observation['report_links']}\n"
                    # ... process observations from other tools similarly)
                    if "competitors" in observation:
                       competitor_analysis = f"Competitor Analysis: {observation.get('competitors')}\n"

                except json.JSONDecodeError as e:
                     logger.error(f"Error parsing observation: {e}, Message: {message.content}") #Log details for debugging
                     
                     industry_overview += f"Unparsed Observation: {message.content}\n"


            elif message.role == "assistant":  
                try:
                   content = json.loads(message.content) 
                   if "brainstormed_use_cases" in content: 
                       use_cases = content["brainstormed_use_cases"]
                       potential_use_cases.extend(use_cases) 

                except (json.JSONDecodeError, KeyError) as e:
                    logger.error(f"Error parsing assistant message: {e}, Message: {message.content}")

            final_answer = {
            "industry_overview": industry_overview,
            "competitor_analysis": competitor_analysis,
            "potential_use_cases": potential_use_cases,

        }
        self.trace("assistant", json.dumps({"answer": final_answer}, indent=2))
            
    def execute(self, query: str) -> Union[Dict[str,Any], str]:
        """
        Executes the agent query workflow.

        Args:
            query (str): The query to process.
        
        Returns:
            Union[Dict, str]: The final answer (as a dictionary) or an error message (string), 
                             or the last message content if no structured answer is found.
        """
        self.query = query
        self.trace(role="user", content=query)
        self.think()

        final_message = self.messages[-1]

        try: 
            message_content = json.loads(final_message.content)

            if final_message.role == "assistant" and "answer" in message_content:
                return message_content["answer"] 
            else:
                return message_content 
        except json.JSONDecodeError: 
           return final_message.content
        
    def ask_gemini(self, prompt: str) -> str:
        """
        Generate text using the Gemini model.

        Args:
            prompt (str): The prompt to generate text from.

        Returns:
            str: The generated text or an error message.
        """
        try:
            # For GenerativeModel, we can pass the prompt string directly
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text
            else:
                error_message = "No response from Gemini"
                logger.error(error_message) 
                return error_message
                
        except Exception as e:  
            error_message = f"Error generating text from Gemini: {e}"
            logger.exception(error_message) 
            return error_message
        
def run(query: str) -> Dict[str, Any]:
    """
    Sets up the agent, registers tools, and executes a query.

    Args:
        query (str): The query to execute.

    Returns:
        Dict[str, Any]: The agent's final answer.
    """
    config = Config()
    gemini_model = GenerativeModel("gemini-pro")
    
    # Create manager first
    manager = Manager(llm=gemini_model, model=gemini_model)
    
    # Register tools with the manager
    manager.register(Name.GOOGLE_SEARCH, "Performs a Google search.", google_search)
    manager.register(Name.INDUSTRY_REPORT, "Finds industry reports", industry_report_tool) 
    manager.register(Name.COMPETITOR_ANALYSIS, "Analyzes competitors", competitor_analysis_tool) 
    manager.register(Name.DATASET_SEARCH, "Searches for datasets", dataset_search_tool) 
    manager.register(Name.PRODUCT_SEARCH, "Searches for products", search_google_products)
    manager.register(Name.GOOGLE_TRENDS, "Searches Google Trends", search_google_trends)
    
    # Then create agent with that manager
    agent = Agent(model=gemini_model, manager=manager)
    
    # Register the same tools with the agent
    for name, tool in manager.tools.items():
        agent.tools[name] = tool

    return agent.execute(query)
    
if __name__ == "__main__":
    query = "Market research for AI use cases in the cosmetics industry, focusing on Sephora."
    final_answer = run(query)
    logger.info(final_answer)