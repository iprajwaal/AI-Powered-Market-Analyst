# /Users/prajwal/Developer/AI-Powered-Market-Analyst/backend/src/react/agent.py
# Modified version with fixes

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
    input: str = Field("", description="Input for the chosen tool.")

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
            log.debug(f"Tool {self.name} returned result")
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
        self.query = ""
        self.max_iterations = 15
        self.current_iteration = 0
        self.min_iterations = 5
        self.template = self._load_template()
        self.tool_sequence = [
            Name.GOOGLE_SEARCH, 
            Name.COMPETITOR_ANALYSIS, 
            Name.INDUSTRY_REPORT, 
            Name.PRODUCT_SEARCH, 
            Name.DATASET_SEARCH
        ]
        self.current_tool_idx = 0

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
    
    def manage_context_window(self, max_tokens=25000):
        """Keep conversation history within token limits by summarizing older messages."""
        # Calculate approximate token count (rough estimation)
        total_tokens = sum(len(m.content.split()) * 1.3 for m in self.messages)
        
        if total_tokens > max_tokens:
            # Summarize oldest messages until within limit
            # Keep the first message (user query) intact
            first_message = self.messages[0] if self.messages else None
            # Summarize the next several messages
            messages_to_summarize = self.messages[1:6] if len(self.messages) > 6 else self.messages[1:]
            
            if messages_to_summarize:
                summary_content = "\n".join([f"{m.role}: {m.content}" for m in messages_to_summarize])
                summary_prompt = f"Summarize these observations in 100 words or less: {summary_content}"
                summary = self.ask_gemini(summary_prompt)
                
                # Replace summarized messages with summary
                self.messages = ([first_message] if first_message else []) + \
                            [Message(role="system", content=f"Summary of earlier observations: {summary}")] + \
                            self.messages[6:] if len(self.messages) > 6 else []
    
    def think(self) -> None:
        """Main reasoning loop with sequential tool usage."""
        self.manage_context_window()
        self.current_iteration += 1
        logger.info(f"Thinking iteration {self.current_iteration}...")
        write_to_file(path=OUTPUT_TRACE_PATH, content=f"\n{'='*50}\nThinking iteration {self.current_iteration}\n{'='*50}\n")

        # Debug info
        logger.debug(f"Available tools: {[t.name for t in self.tools.values()]}")
        
        if self.current_iteration >= self.max_iterations:
            logger.info("Max iterations reached. Stopping!")
            self.generate_final_answer()
            return

        # Get the tool in the predefined sequence
        if self.current_tool_idx < len(self.tool_sequence):
            current_tool_name = self.tool_sequence[self.current_tool_idx]
            
            # Make sure the tool exists
            if current_tool_name not in self.tools:
                logger.error(f"Tool {current_tool_name} not found in registered tools!")
                self.trace("system", f"Error: Tool {current_tool_name} not found")
                self.current_tool_idx += 1  # Skip to next tool
                self.think()
                return
            
            logger.info(f"Using tool {current_tool_name} (iteration {self.current_iteration})")
            
            # Use the manager to force tool selection
            try:
                choice = self.manager.force_tool_selection(current_tool_name, self.query)
                self.trace("assistant", f"I'll use the {choice.name} tool: {choice.reason}")
                
                # Increment the tool index for next iteration
                self.current_tool_idx += 1
                
                # Execute the tool
                self.act(choice)
                
            except Exception as e:
                logger.error(f"Error selecting or using tool {current_tool_name}: {e}")
                self.trace("system", f"Error using tool {current_tool_name}: {str(e)}")
                self.current_tool_idx += 1  # Skip to next tool
                self.think()
        else:
            # If we've gone through all tools, generate the final answer
            logger.info("All tools in sequence have been used. Generating final answer.")
            self.generate_final_answer()

    def act(self, choice: Choice) -> None:
        """
        Executes the chosen tool and processes the result.
        
        Args:
            choice (Choice): The choice of tool to act on.
            
        Returns:
            None
        """
        tool_name_str = choice.name
        
        # Get the full Name enum if possible
        try:
            if isinstance(tool_name_str, str):
                try:
                    tool_name = Name[tool_name_str.upper()]
                except KeyError:
                    # If the exact name doesn't match, try to find closest match
                    for name in Name:
                        if tool_name_str.upper() in name.name:
                            tool_name = name
                            break
                    else:
                        raise ValueError(f"Cannot map {tool_name_str} to any tool name")
            else:
                tool_name = tool_name_str
        except Exception as e:
            logger.error(f"Invalid tool name: {tool_name_str}, Error: {e}")
            self.trace("system", f"Error: Invalid tool name {tool_name_str}")
            self.think()  # Continue to next tool in sequence
            return
        
        logger.info(f"Executing tool: {tool_name}")
        
        # Get the tool instance
        tool = self.tools.get(tool_name)
        
        if not tool:
            logger.error(f"No tool registered for: {tool_name}")
            self.trace("system", f"Error: Tool {tool_name} not found")
            self.think()  # Continue to next tool in sequence
            return
        
        # Log what we're doing
        self.trace("assistant", f"Using tool: {tool_name}")
        
        try:
            query = choice.input or self.query  # Use provided input or fall back to original query
            result = tool.use(query)
            
            # Process and store the result
            if isinstance(result, tuple):
                status_code, error_message = result
                observation = {"error": f"Tool {tool_name} failed with status {status_code}: {error_message}"}
            
            elif isinstance(result, str) and len(result) > 10000:
                # For large string results, truncate and summarize
                logger.info(f"Large result from {tool_name}: {len(result)} chars. Summarizing...")
                summary_prompt = f"Summarize this information in 500 words or less: {result[:15000]}"
                summary = self.ask_gemini(summary_prompt)
                
                observation = {
                    "result_summary": summary,
                    "result_length": len(result),
                    "result_sample": result[:1000] + "..."  # Include just a sample
                }
            elif isinstance(result, dict):
                observation = result
            elif isinstance(result, str):
                try:
                    observation = json.loads(result)
                except json.JSONDecodeError:
                    observation = {"result": result}  # Store as plain text if not JSON
            else:
                observation = {"error": f"Unexpected tool output type: {type(result)}", "raw_output": str(result)}
            
            # Create a more concise version for logging
            log_observation = {
                "tool": str(tool_name),
                "result_type": type(result).__name__,
                "result_length": len(str(result)) if result else 0
            }
            observation_message = f"Observation from {tool_name}: {json.dumps(log_observation, indent=2)}"
            self.trace("system", observation_message) 
            
            # Store the full observation in the history
            self.messages.append(Message(role="system", content=json.dumps(observation)))
            
            # Give the assistant a chance to interpret the results
            analysis_prompt = f"""
            You are a market research assistant analyzing information from different tools.
            
            Tool used: {tool_name}
            Query: {query}
            
            Tool result: {json.dumps(observation, indent=2)}
            
            Please provide a brief analysis of this information:
            1. What are the key insights?
            2. How does this inform our understanding of the market?
            3. What should we explore next?
            
            Keep your response under 200 words.
            """
            
            analysis = self.ask_gemini(analysis_prompt)
            
            # Add the analysis to the conversation
            self.trace("assistant", analysis)
            
            # Continue the thinking process
            self.think()
            
        except Exception as e:
            error_message = f"Unexpected error using {tool_name}: {e}"
            logger.exception(error_message)
            self.trace("system", error_message)
            self.think()  # Continue to next tool

    def generate_final_answer(self) -> None:
        """Generates the final structured answer based on all collected data."""

        # First, collect all the information from previous tool calls
        industry_overview = ""
        competitor_analysis = ""
        potential_use_cases = []
        product_info = ""
        dataset_info = ""

        for message in self.messages:
            if message.role == "system": 
                try:
                    # Try to parse as JSON
                    observation = None
                    try:
                        observation = json.loads(message.content)
                    except json.JSONDecodeError:
                        # If not valid JSON, use as plain text
                        observation = {"plain_text": message.content}
                    
                    # Extract information based on what's available
                    if isinstance(observation, dict):
                        # Process industry reports
                        if "report_links" in observation:
                            industry_overview += f"Industry Report Links: {observation['report_links']}\n"
                        if "report_summary" in observation:
                            industry_overview += f"Report Summary: {observation['report_summary']}\n"
                        if "result_summary" in observation:
                            industry_overview += f"Summary: {observation['result_summary']}\n"
                            
                        # Process competitor analysis
                        if "competitors" in observation:
                            competitor_analysis += f"Competitor Analysis: {observation.get('competitors')}\n"
                        if "competition" in observation:
                            competitor_analysis += f"Competition: {observation.get('competition')}\n"
                            
                        # Process product information
                        if "products" in observation:
                            product_info += f"Products: {observation.get('products')}\n"
                        
                        # Process dataset information
                        if "datasets" in observation:
                            dataset_info += f"Datasets: {observation.get('datasets')}\n"
                        
                        # Process plain text (add to industry overview as fallback)
                        if "plain_text" in observation:
                            industry_overview += f"{observation.get('plain_text')}\n"

                except Exception as e:
                    logger.error(f"Error processing observation: {e}, Message: {message.content}")
                    industry_overview += f"Unparsed Observation: {message.content[:100]}\n"
            
            # Extract brainstormed use cases if present
            elif message.role == "assistant":  
                try:
                    # Try to parse as JSON
                    content = None
                    try:
                        content = json.loads(message.content)
                    except json.JSONDecodeError:
                        # Not a JSON message, skip
                        pass
                        
                    if isinstance(content, dict) and "brainstormed_use_cases" in content:
                        use_cases = content["brainstormed_use_cases"]
                        potential_use_cases.extend(use_cases)
                except Exception as e:
                    logger.error(f"Error parsing assistant message: {e}")

        # If we don't have any use cases yet, generate them
        if not potential_use_cases:
            # Summarize all collected information
            all_info = f"""
            Industry Overview: {industry_overview}
            Competitor Analysis: {competitor_analysis}
            Product Information: {product_info}
            Dataset Information: {dataset_info}
            """
            
            # Generate use cases based on collected information
            brainstorm_prompt = f"""
            Based on the following information about the cosmetics industry and Sephora, 
            generate 5 innovative AI/GenAI use cases:
            
            {all_info}
            
            For each use case, provide:
            1. A title
            2. A detailed description
            3. Potential benefits
            4. Implementation challenges
            
            Format as a JSON array of objects, each with a "use_case" field.
            """
            
            try:
                use_cases_json = self.ask_gemini_with_backoff(brainstorm_prompt)
                use_cases = json.loads(use_cases_json)
                potential_use_cases = use_cases
            except (json.JSONDecodeError, Exception) as e:
                logger.error(f"Error generating use cases: {e}")
                # Create a simple use case as fallback
                potential_use_cases = [
                    {"use_case": "AI-powered skin analysis for personalized product recommendations", "resources": []}
                ]

        # Now generate the final answer
        final_answer = {
            "industry_overview": industry_overview or "Information about the cosmetics industry focusing on Sephora, including market trends and opportunities for AI integration.",
            "competitor_analysis": competitor_analysis or "Analysis of Sephora's competitors in the cosmetics space and their AI initiatives.",
            "potential_use_cases": potential_use_cases or [
                {"use_case": "AI-powered virtual makeup try-on for Sephora customers", "resources": []}
            ],
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
        
        # Reset iteration counter and tool index
        self.current_iteration = 0
        self.current_tool_idx = 0
        
        # Start the thinking loop
        self.think()

        # Get the final message (should contain the answer)
        if not self.messages:
            return {"error": "No messages generated during execution"}
            
        final_message = self.messages[-1]

        try: 
            # Try to parse the final message as JSON
            message_content = json.loads(final_message.content)

            if final_message.role == "assistant" and "answer" in message_content:
                return message_content["answer"] 
            else:
                return message_content 
        except json.JSONDecodeError: 
           return final_message.content
        
    def ask_gemini_with_backoff(self, prompt: str, max_retries=5) -> str:
        """
        Generate text using the Gemini model with exponential backoff for rate limits.
        
        Args:
            prompt (str): The prompt to generate text from.
            max_retries (int): Maximum number of retry attempts.
            
        Returns:
            str: The generated text or an error message.
        """
        import time
        import random
        
        retry = 0
        while retry < max_retries:
            try:
                # For GenerativeModel, we can pass the prompt string directly
                response = self.model.generate_content(prompt)
                
                if response and hasattr(response, 'text') and response.text:
                    return response.text
                else:
                    error_message = "No response from Gemini"
                    logger.error(error_message) 
                    return error_message
                    
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):  # Rate limit error
                    wait_time = (2 ** retry) + random.uniform(0, 1)
                    logger.warning(f"Rate limited. Retrying in {wait_time}s")
                    time.sleep(wait_time)
                    retry += 1
                else:
                    error_message = f"Error generating text from Gemini: {e}"
                    logger.exception(error_message) 
                    return error_message
        
        return "Failed after max retries due to rate limiting"
        
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
            
            if response and hasattr(response, 'text') and response.text:
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