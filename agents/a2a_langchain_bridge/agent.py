# adk-demo/agents/a2a_langchain_bridge/agent.py
import logging
from google.adk.agents import Agent
from google.genai.types import Content, Part
from .tools import langchain_a2a_tool # Import the specific tool instance

logger = logging.getLogger(__name__)

class A2ALangchainBridgeAgent(Agent):
    """
    An ADK Agent that acts as a bridge to an external LangGraph agent
    using the LangchainA2ATool (which implements A2A JSON-RPC).
    """
    # Define a unique name for this agent
    name: str = "A2ALangchainBridgeAgent" # Consider renaming if connecting specifically to LangGraph, e.g., A2ALangGraphBridgeAgent
    # Description shown in logs or potentially UI
    description: str = "Relays user queries to an external LangGraph agent via A2A JSON-RPC."

    def __init__(self):
        """Initializes the agent with the necessary tool."""
        # The list of tools this agent can directly use
        super().__init__(tools=[langchain_a2a_tool])
        logger.info(f"Initialized {self.name} with tool: {langchain_a2a_tool.name}") # .name should now work correctly

    def __call__(self, message: Content) -> Content | None:
        """
        Processes the user message by invoking the LangChain A2A tool.

        Args:
            message: The input Content object from the ADK runner, containing the user's query.

        Returns:
            A Content object containing the response from the LangGraph agent,
            or None if the agent cannot handle the request.
        """
        # Extract the primary text part of the message
        if not message.parts or not hasattr(message.parts[0], 'text'):
             logger.warning(f"{self.name} received a message with no text part.")
             return None

        query_text = message.parts[0].text
        logger.info(f"Received message for {self.name}: '{query_text}'")

        # Pass the original text to the tool
        query_for_tool = query_text

        if not query_for_tool:
            logger.warning(f"{self.name} received an empty query.")
            return Content(role=self.name, parts=[Part(text="Please provide a query for the LangGraph agent.")])

        # Call the tool instance directly
        # ADK expects the tool function to be called with keyword arguments
        tool_response_dict = self.tools[0](query=query_for_tool) # self.tools[0] refers to langchain_a2a_tool

        # Extract the response text or error message from the dictionary
        if tool_response_dict.get("status") == "success":
            response_text = tool_response_dict.get("result", "Agent returned success but no result text.")
        else:
            response_text = tool_response_dict.get("error", "An unknown error occurred in the A2A tool.")


        # Wrap the tool's string response in an ADK Content object
        response_content = Content(
            role=self.name, # Role should be the agent's name
            parts=[Part(text=response_text)]
        )
        logger.info(f"Returning response from {self.name}.")
        logger.debug(f"Response content: {response_text}")
        return response_content

# Instantiate the agent for import by the MetaAgent
a2a_langchain_bridge_agent = A2ALangchainBridgeAgent()