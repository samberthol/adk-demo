# ui/app.py
import streamlit as st
import logging
import time
import os
import sys
from pathlib import Path
import json # For potential message parsing if needed later

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import ADK components
try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    # Import Content and Part for constructing messages
    from google.genai.types import Content, Part
except ImportError as e:
    st.error(f"Failed to import agent modules or ADK components. Check path and file existence. Error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

# --- Constants ---
APP_NAME = "gcp_multi_agent_demo_streamlit" # Runner uses this
USER_ID = f"st_user_{APP_NAME}" # Define a consistent User ID for this Streamlit app instance
ADK_SESSION_STATE_KEY = f'adk_session_id_{APP_NAME}'

# --- ADK Initialization ---
# Use Streamlit session state to store ADK components to persist across reruns
# Use @st.cache_resource for components that should persist across reruns for all users of this instance
@st.cache_resource
def get_session_service():
    logger.info("--- ADK Init: Instantiating InMemorySessionService (_SessionService)... ---")
    return InMemorySessionService()

@st.cache_resource
def get_adk_runner(_session_service): # Pass service as argument
    logger.info("--- ADK Init: Instantiating Runner (_Runner)... ---")
    return Runner(
        agent=meta_agent,
        app_name=APP_NAME, # Pass app_name to Runner
        session_service=_session_service # Use the cached service
    )

session_service = get_session_service()
adk_runner = get_adk_runner(session_service)

# --- Helper Functions for ADK Interaction ---

# Apply nest_asyncio once if needed
if 'nest_asyncio_applied' not in st.session_state:
     try:
         import nest_asyncio
         nest_asyncio.apply()
         st.session_state.nest_asyncio_applied = True
         logger.info("nest_asyncio applied.")
     except ImportError:
         logger.warning("nest_asyncio not found.")
         st.session_state.nest_asyncio_applied = False
     except RuntimeError as e:
         if "cannot apply nest_asyncio" in str(e):
            logger.info("nest_asyncio already applied or cannot be applied again.")
            st.session_state.nest_asyncio_applied = True
         else:
            logger.error(f"Error applying nest_asyncio: {e}")
            st.session_state.nest_asyncio_applied = False


def get_or_create_adk_session_id() -> str:
    """
    Generates or retrieves a unique ADK session ID stored in Streamlit's session state.
    Does NOT interact with the session service directly, assumes Runner handles it.
    """
    if ADK_SESSION_STATE_KEY not in st.session_state:
        # Generate a new session ID if none exists for this Streamlit session
        session_id = f"st_session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_STATE_KEY] = session_id
        logger.info(f"--- Streamlit Session: Generated new ADK session ID: {session_id} ---")
    else:
        # Reuse the existing session ID from Streamlit's state
        session_id = st.session_state[ADK_SESSION_STATE_KEY]
        logger.info(f"--- Streamlit Session: Reusing ADK session ID from state: {session_id} ---")
    return session_id


def run_adk_turn_sync(runner: Runner, user_id: str, session_id: str, user_message_text: str) -> str:
    """Runs a single turn with the ADK agent synchronously."""
    # Runner requires user_id and session_id for context
    logger.info(f"--- Streamlit ADK Run Sync: Session {session_id}, User: {user_id}, Query: '{user_message_text[:100]}...' ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"

    try:
        # Use the synchronous runner method
        # Assume the Runner interacts with the InMemorySessionService implicitly
        # using the provided session_id and user_id.
        for event in runner.run(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                    logger.info(f"--- Streamlit ADK Run Sync: Session {session_id} - Final response received.")
                else:
                    final_response_text = "[Agent finished with no text output]"
                    logger.warning(f"--- Streamlit ADK Run Sync: Session {session_id} - Final event has no text content.")
                break # Exit loop once final response is found
    except Exception as e:
        logger.exception(f"--- Streamlit ADK Run Sync: Session {session_id} - !! EXCEPTION during agent execution:")
        final_response_text = f"Sorry, an error occurred while processing your request: {e}"

    logger.info(f"--- Streamlit ADK Run Sync: Session {session_id} - Turn complete. Response: '{final_response_text[:100]}...' ---")
    return final_response_text


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("GCP Multi-Agent Demo (Text Chat)")

# Initialize chat history in session state if it doesn't exist
message_history_key = f"messages_{APP_NAME}"
if message_history_key not in st.session_state:
    st.session_state[message_history_key] = [{"role": "assistant", "content": "Hello! How can I help you manage GCP resources or query data today?"}]


# Get or create the ADK session ID for this user/session
# This now only manages the ID string in Streamlit's state
try:
     current_adk_session_id = get_or_create_adk_session_id()
     # Display success only if ID is obtained
     st.sidebar.success(f"ADK Session Active\nID: ...{current_adk_session_id[-12:]}", icon="✅")
except Exception as e:
     # Catch potential errors during ID generation/retrieval (though less likely now)
     st.error(f"Fatal error managing session ID: {e}. Please refresh.")
     logger.exception("Fatal error during ADK session ID management:")
     st.stop()


# Display chat messages from history on app rerun
for message in st.session_state[message_history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to do? (e.g., 'list vms', 'run query select * from `project.dataset.table` limit 5')"):
    # Ensure session ID was initialized before proceeding
    if not current_adk_session_id:
         st.error("Agent session ID could not be established. Cannot process request.")
    else:
        # Add user message to chat history
        st.session_state[message_history_key].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display thinking indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            # Run the ADK turn synchronously
            try:
                # Call the synchronous ADK turn function directly
                # Pass the cached runner, the consistent USER_ID, and the current session_id
                agent_response = run_adk_turn_sync(adk_runner, USER_ID, current_adk_session_id, prompt)
                message_placeholder.markdown(agent_response) # Update placeholder with actual response
                st.session_state[message_history_key].append({"role": "assistant", "content": agent_response}) # Add response to history

            except Exception as e:
                logger.exception("Error running ADK turn from Streamlit input:")
                error_msg = f"An error occurred: {e}"
                message_placeholder.markdown(error_msg)
                st.session_state[message_history_key].append({"role": "assistant", "content": error_msg})

# Add a button to clear the chat history and session ID from Streamlit state
st.sidebar.title("Session Control")
if st.sidebar.button("Clear Chat & Reset Session ID"):
    logger.info(f"Clearing chat history for user {USER_ID}, ADK session {current_adk_session_id}")
    st.session_state[message_history_key] = [] # Clear messages
    # Remove the session ID key from Streamlit state, forcing regeneration on next run
    if ADK_SESSION_STATE_KEY in st.session_state:
        del st.session_state[ADK_SESSION_STATE_KEY]
        logger.info(f"Removed ADK session key '{ADK_SESSION_STATE_KEY}' from Streamlit state.")
    # Rerun to reflect the cleared history and force ID regeneration
    st.rerun()

# Optional Debug button to clear entire state (use with caution)
# if st.sidebar.button("Clear Full Session State (Debug)"):
#     logger.warning("Clearing full Streamlit session state.")
#     keys_to_keep = ['query_params'] # Example
#     kept_values = {k: st.session_state[k] for k in keys_to_keep if k in st.session_state}
#     st.session_state.clear()
#     for k, v in kept_values.items(): st.session_state[k] = v
#     st.rerun()


# --- Sidebar Information ---
st.sidebar.divider()
st.sidebar.header("Agent Details")
# Use the cached runner instance to get agent name
st.sidebar.caption(f"**Agent Name:** `{adk_runner.agent.name if adk_runner and adk_runner.agent else 'N/A'}`")
st.sidebar.caption(f"**App Name:** `{APP_NAME}`")
st.sidebar.caption(f"**User ID:** `{USER_ID}`")
st.sidebar.caption(f"**Current Session ID:** `{st.session_state.get(ADK_SESSION_STATE_KEY, 'N/A')}`")