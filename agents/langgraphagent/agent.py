# adk-demo/agents/langgraphagent/agent.py
import logging
from google.adk.agents import Agent
# Import the concrete session service implementation for type hinting
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from .tools import langgraph_currency_tool

logger = logging.getLogger(__name__)

A2A_SESSION_ID_KEY = 'langgraph_a2a_session_id'

class A2ALangGraphCurrencyAgent(Agent):
    """
    ADK Agent bridge to an external LangGraph Currency Agent via A2A JSON-RPC.
    Manages the A2A session ID across turns using the ADK session service.
    """
    name: str = "A2ALangGraphCurrencyAgent"
    description: str = "Relays queries to the external LangGraph Currency agent."

    # Updated type hint to use the concrete class
    def __init__(self, session_service: InMemorySessionService):
        """
        Initializes the agent with the tool and session service.

        Args:
            session_service: The ADK session service (InMemorySessionService instance).
        """
        super().__init__(tools=[langgraph_currency_tool])
        self._session_service = session_service
        logger.info(f"Initialized {self.name} with tool: {langgraph_currency_tool.name}")

    def __call__(self, message: Content, session_id: str | None = None, user_id: str | None = None) -> Content | None:
        """
        Processes user message, invokes tool, manages A2A session ID via ADK session.
        """
        if not session_id or not user_id:
            logger.error(f"{self.name} requires ADK session_id and user_id.")
            return Content(role=self.name, parts=[Part(text="Internal Error: ADK session context missing.")])

        if not message.parts or not hasattr(message.parts[0], 'text'):
             logger.warning(f"{self.name} received message with no text part.")
             return Content(role=self.name, parts=[Part(text="Please provide query as text.")])

        query_text = message.parts[0].text
        if not query_text:
            logger.warning(f"{self.name} received empty query.")
            return Content(role=self.name, parts=[Part(text="Please provide a query.")])

        logger.info(f"Received message for {self.name} (ADK Session: {session_id}): '{query_text}'")

        # Retrieve A2A Session ID
        a2a_session_id: str | None = None
        try:
            # Use self.app_name if set by Runner, otherwise fallback if needed
            app_name = getattr(self, 'app_name', 'UNKNOWN_APP')
            session_data = self._session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            if session_data and session_data.state:
                a2a_session_id = session_data.state.get(A2A_SESSION_ID_KEY)
                logger.info(f"Retrieved A2A session ID: {a2a_session_id}" if a2a_session_id else "No prior A2A session ID found.")
            else:
                 logger.warning(f"No state found for ADK session {session_id}.")
        except Exception as e:
            logger.exception(f"Error retrieving ADK session state for {session_id}:")
            a2a_session_id = None

        # Call Tool
        tool_response_dict = self.tools[0](query=query_text, session_id=a2a_session_id)

        # Process Response
        response_text = "Error processing request."
        returned_a2a_session_id = tool_response_dict.get("session_id")
        final_state = tool_response_dict.get("state", "unknown")

        if tool_response_dict.get("status") == "success":
            response_text = tool_response_dict.get("result", "Agent returned success but no result.")
        else:
            response_text = tool_response_dict.get("error", "Unknown error from tool.")
            logger.error(f"{self.name} tool call failed. State: {final_state}, Error: {response_text}")

        # Store/Update A2A Session ID
        if returned_a2a_session_id and returned_a2a_session_id != a2a_session_id:
            try:
                app_name = getattr(self, 'app_name', 'UNKNOWN_APP')
                current_session_data = self._session_service.get_session(
                    app_name=app_name, user_id=user_id, session_id=session_id
                )
                new_state = current_session_data.state if current_session_data and current_session_data.state else {}
                new_state[A2A_SESSION_ID_KEY] = returned_a2a_session_id
                self._session_service.update_session(
                    app_name=app_name, user_id=user_id, session_id=session_id, state=new_state
                )
                logger.info(f"Stored/Updated A2A session ID {returned_a2a_session_id} in ADK session {session_id}")
            except Exception as e:
                logger.exception(f"Error updating ADK session state for {session_id}:")

        # Return Response
        response_content = Content(role=self.name, parts=[Part(text=response_text)])
        logger.info(f"Returning response from {self.name} (ADK Session: {session_id}).")
        return response_content
