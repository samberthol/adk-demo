# adk-demo/agents/langgraphagent/agent.py
import logging
from google.adk.agents import Agent
from google.adk.sessions import SessionService # Needed for session management
from google.genai.types import Content, Part
from .tools import langgraph_currency_tool # Import the specific tool instance

logger = logging.getLogger(__name__)

# Define a key to store the A2A session ID within the ADK session state
A2A_SESSION_ID_KEY = 'langgraph_a2a_session_id'

class A2ALangGraphCurrencyAgent(Agent):
    """
    An ADK Agent that acts as a bridge to an external LangGraph Currency Agent
    using the langgraph_currency_tool (which implements A2A JSON-RPC 'tasks/send').
    It manages the A2A session ID across turns within the ADK session.
    """
    # Define a unique name for this agent
    name: str = "A2ALangGraphCurrencyAgent"
    # Description shown in logs or potentially UI
    description: str = "Relays user queries to the external LangGraph Currency agent via A2A JSON-RPC and handles responses."

    def __init__(self, session_service: SessionService):
        """
        Initializes the agent with the necessary tool and session service.

        Args:
            session_service: The ADK session service to store A2A session IDs.
        """
        # The list of tools this agent can directly use
        super().__init__(tools=[langgraph_currency_tool])
        self._session_service = session_service # Store session service instance
        logger.info(f"Initialized {self.name} with tool: {langgraph_currency_tool.name} and session service.")

    def __call__(self, message: Content, session_id: str | None = None, user_id: str | None = None) -> Content | None:
        """
        Processes the user message by invoking the LangGraph Currency A2A tool.
        Retrieves and stores the A2A session ID using the ADK session service.

        Args:
            message: The input Content object from the ADK runner.
            session_id: The current ADK session ID (provided by the runner).
            user_id: The current user ID (provided by the runner).

        Returns:
            A Content object containing the response from the LangGraph agent,
            or None if the agent cannot handle the request.
        """
        # --- Input Validation ---
        if not session_id or not user_id:
            logger.error(f"{self.name} requires ADK session_id and user_id to manage A2A session state.")
            return Content(role=self.name, parts=[Part(text="Internal Error: ADK session context missing.")])

        if not message.parts or not hasattr(message.parts[0], 'text'):
             logger.warning(f"{self.name} received a message with no text part.")
             # Optionally return a message asking for text input
             return Content(role=self.name, parts=[Part(text="Please provide your currency query as text.")])

        query_text = message.parts[0].text
        if not query_text:
            logger.warning(f"{self.name} received an empty query.")
            return Content(role=self.name, parts=[Part(text="Please provide a query for the currency agent.")])

        logger.info(f"Received message for {self.name} (ADK Session: {session_id}): '{query_text}'")

        # --- Retrieve A2A Session ID from ADK Session State ---
        a2a_session_id: str | None = None
        try:
            session_data = self._session_service.get_session(
                app_name=self.app_name, # Assuming app_name is set by Runner
                user_id=user_id,
                session_id=session_id
            )
            if session_data and session_data.state:
                a2a_session_id = session_data.state.get(A2A_SESSION_ID_KEY)
                if a2a_session_id:
                    logger.info(f"Retrieved existing A2A session ID: {a2a_session_id} from ADK session {session_id}")
                else:
                    logger.info(f"No A2A session ID found in ADK session {session_id}. Tool will start a new one.")
            else:
                 logger.warning(f"Could not retrieve state for ADK session {session_id}. Starting new A2A session.")

        except Exception as e:
            logger.exception(f"Error retrieving ADK session state for {session_id}:")
            # Proceed without a2a_session_id, the tool will start a new one
            a2a_session_id = None


        # --- Call the Tool ---
        # The tool function expects keyword arguments. Pass the retrieved a2a_session_id.
        # self.tools[0] refers to langgraph_currency_tool
        tool_response_dict = self.tools[0](query=query_text, session_id=a2a_session_id)

        # --- Process Tool Response ---
        response_text = "An unexpected error occurred in the A2A tool."
        returned_a2a_session_id = tool_response_dict.get("session_id")
        final_state = tool_response_dict.get("state", "unknown")

        if tool_response_dict.get("status") == "success":
            response_text = tool_response_dict.get("result", "Agent returned success but no result text.")
            if final_state == "input-required":
                logger.info(f"{self.name} received 'input-required' state. Returning agent's prompt.")
                # Response text already contains the agent's prompt in this case
            elif final_state == "completed":
                 logger.info(f"{self.name} received 'completed' state.")
            else:
                 logger.warning(f"{self.name} received success status but unexpected final state: {final_state}")

        else: # status == "error"
            response_text = tool_response_dict.get("error", "An unknown error occurred in the A2A tool.")
            logger.error(f"{self.name} tool call failed. Error: {response_text}")
            # Optionally clear the A2A session ID on error? Depends on desired behavior.
            # returned_a2a_session_id = None


        # --- Store/Update A2A Session ID in ADK Session State ---
        if returned_a2a_session_id and returned_a2a_session_id != a2a_session_id:
            try:
                # Get current state first to avoid overwriting other state variables
                current_session_data = self._session_service.get_session(
                    app_name=self.app_name, user_id=user_id, session_id=session_id
                )
                new_state = current_session_data.state if current_session_data and current_session_data.state else {}
                new_state[A2A_SESSION_ID_KEY] = returned_a2a_session_id

                self._session_service.update_session(
                    app_name=self.app_name,
                    user_id=user_id,
                    session_id=session_id,
                    state=new_state
                )
                logger.info(f"Stored/Updated A2A session ID {returned_a2a_session_id} in ADK session {session_id}")
            except Exception as e:
                logger.exception(f"Error updating ADK session state for {session_id} with A2A session ID:")
                # Continue, but multi-turn might break

        # --- Return Response ---
        response_content = Content(
            role=self.name, # Role should be the agent's name
            parts=[Part(text=response_text)]
        )
        logger.info(f"Returning response from {self.name} (ADK Session: {session_id}).")
        logger.debug(f"Response content: {response_text}")
        return response_content

# Note: Instantiation needs the session_service, which is typically available
# within the Runner setup (e.g., in ui/app.py or wherever the MetaAgent is created).
# So, we don't instantiate it here directly. It will be instantiated when the
# MetaAgent is created, passing the session_service.
# Example (in meta/agent.py or ui/app.py):
# from google.adk.sessions import InMemorySessionService
# from agents.langgraphagent.agent import A2ALangGraphCurrencyAgent
# session_service = InMemorySessionService() # Or get from st.session_state
# a2a_langgraph_currency_agent = A2ALangGraphCurrencyAgent(session_service=session_service)
# # Then add a2a_langgraph_currency_agent to MetaAgent's sub_agents list
