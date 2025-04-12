# ui/app.py
import streamlit as st
import os
import sys
from pathlib import Path
import asyncio
import uuid
import time
import logging

# Add project root to Python path to allow importing agent modules
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary ADK components and the agent
try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    # Use InMemorySessionService as requested
    from google.adk.sessions import InMemorySessionService, Session
    from google.genai.types import Content, Part
except ImportError as e:
    st.error(f"Failed to import agent modules or ADK components. Ensure project structure and requirements are correct. Error: {e}")
    st.stop()

# Configure logging level (ERROR hides most ADK logs, INFO shows more detail)
logging.basicConfig(level=logging.INFO)

APP_NAME = "gcp_multi_agent_demo"
USER_ID = f"streamlit_user_{APP_NAME}"
ADK_SESSION_STATE_KEY = f'adk_session_id_{APP_NAME}'


# --------------------------------------------------------------------------
# ADK Initialization Logic (inspired by the article)
# --------------------------------------------------------------------------
@st.cache_resource
def initialize_adk():
    """
    Initializes the ADK Runner and InMemorySessionService for the application.
    Manages the unique ADK session ID within the Streamlit session state.
    Includes check and recreation logic for InMemorySessionService.

    Returns:
        tuple: (Runner instance, active ADK session ID)
    """
    logging.info("--- ADK Init: Attempting to initialize Runner and Session Service... ---")
    session_service = InMemorySessionService()
    logging.info("--- ADK Init: InMemorySessionService instantiated. ---")

    runner = Runner(
        agent=meta_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    logging.info(f"--- ADK Init: Runner instantiated for agent '{meta_agent.name}'. ---")

    # Manage ADK session ID within Streamlit's session state
    if ADK_SESSION_STATE_KEY not in st.session_state:
        session_id = f"session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_STATE_KEY] = session_id
        logging.info(f"--- ADK Init: Generated new ADK session ID: {session_id} ---")
        try:
            # Create the session record in the service (assuming sync or handles internally)
            # Note: Session service methods might ideally be async, requiring `await`
            # and an async `initialize_adk`. Using sync calls based on article's apparent pattern.
            session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state={}
            )
            logging.info("--- ADK Init: Successfully created new session in ADK SessionService.")
        except Exception as e:
            logging.exception("--- ADK Init: FATAL ERROR - Could not create initial session in ADK SessionService:")
            raise # Stop execution if session can't be created
    else:
        session_id = st.session_state[ADK_SESSION_STATE_KEY]
        logging.info(f"--- ADK Init: Reusing existing ADK session ID from Streamlit state: {session_id} ---")
        try:
            # **Important Check for InMemorySessionService**:
            # Check if the session actually exists in the service's memory.
            # Assuming get_session might return None or raise error if not found (adjust based on actual ADK behavior)
            existing_session = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

            if not existing_session:
                logging.warning(f"--- ADK Init: Session {session_id} not found in InMemorySessionService memory (likely due to script restart). Recreating session. State will be lost. ---")
                try:
                    # Recreate the session record in the service. State resets to empty.
                    session_service.create_session(
                        app_name=APP_NAME,
                        user_id=USER_ID,
                        session_id=session_id,
                        state={}
                    )
                    logging.info(f"--- ADK Init: Successfully recreated session {session_id} in ADK SessionService.")
                except Exception as e_recreate:
                    logging.exception(f"--- ADK Init: ERROR - Could not recreate missing session {session_id} in ADK SessionService:")
                    # Depending on requirements, raise error or proceed carefully
            else:
                 logging.info(f"--- ADK Init: Session {session_id} successfully found in SessionService memory.")

        except Exception as e_get:
             # Handle potential errors during get_session itself
             logging.error(f"--- ADK Init: Error trying to get session {session_id} from service: {e_get} ---")
             # Decide how to proceed - maybe attempt recreation or raise? For now, log and continue.


    logging.info(f"--- ADK Init: Initialization sequence complete. Runner is ready. Active Session ID: {session_id} ---")
    return runner, session_id


# --------------------------------------------------------------------------
# Async Runner Function
# --------------------------------------------------------------------------
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Asynchronously executes one turn of the ADK agent conversation.
    """
    logging.info(f"\n--- ADK Run: Starting async execution for session {session_id} ---")
    logging.info(f"--- ADK Run: Processing User Query (truncated): '{user_message_text[:150]}...' ---")
    content = Content(
        role='user',
        parts=[Part(text=user_message_text)]
    )
    final_response_text = "[Agent encountered an issue and did not produce a final response]"
    start_time = time.time()
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                logging.info("--- ADK Run: Final response event received.")
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                else:
                    final_response_text = "[Agent finished but produced no text output]"
                    logging.warning(f"--- ADK Run: Final event received, but no text content found. Event: {event}")
                break
            else:
                 pass # Ignoring intermediate events for now

    except Exception as e:
        logging.exception("--- ADK Run: !! EXCEPTION during agent execution:")
        final_response_text = f"Sorry, an error occurred while processing your request. Please check the logs or try again later. (Error: {e})"

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"--- ADK Run: Turn execution completed in {duration:.2f} seconds.")
    logging.info(f"--- ADK Run: Final Response (truncated): '{final_response_text[:150]}...' ---")
    return final_response_text

# --------------------------------------------------------------------------
# Sync Wrapper for Streamlit
# --------------------------------------------------------------------------
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Synchronous wrapper that executes the asynchronous run_adk_async function.
    """
    # Handle potential asyncio event loop issues within Streamlit
    try:
        return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
    except RuntimeError as e:
        if "cannot run nested" in str(e):
            # If nested loop error, try using nest_asyncio
            logging.warning("Asyncio nested loop detected. Applying nest_asyncio.")
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
        else:
            logging.exception("Unhandled RuntimeError during asyncio.run:")
            raise e

# --------------------------------------------------------------------------
# Streamlit User Interface Setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="ADK GCP Agent", layout="wide")
st.title("💬 GCP Multi-Agent Demo")
st.caption("🚀 Powered by Google ADK & Cloud Run")

# --- Initialize ADK Runner and Session ---
try:
    adk_runner, current_adk_session_id = initialize_adk()
    st.sidebar.success(f"ADK Initialized\nSession: ...{current_adk_session_id[-12:]}", icon="✅")
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize the ADK Runner or Session Service: {e}", icon="❌")
    st.error("Please check the terminal logs for more details, ensure your API key is valid (if needed), and restart the application.")
    logging.exception("Critical ADK Initialization failed in Streamlit UI context.")
    st.stop() # Stop the app if ADK fails to initialize

# --- Chat Interface Implementation ---
message_history_key = f"messages_{APP_NAME}"
if message_history_key not in st.session_state:
    st.session_state[message_history_key] = [{"role": "assistant", "content": "Hello! How can I help you manage GCP resources or query data today?"}]

for message in st.session_state[message_history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=False)

if prompt := st.chat_input("Enter your request (e.g., 'Create a VM named my-vm in us-central1' or 'Query dataset X')"):
    st.session_state[message_history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt, unsafe_allow_html=False)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Agent is thinking..."):
            try:
                agent_response = run_adk_sync(adk_runner, current_adk_session_id, USER_ID, prompt)
                message_placeholder.markdown(agent_response, unsafe_allow_html=False)
            except Exception as e:
                error_msg = f"Sorry, an error occurred while getting the agent response: {e}"
                st.error(error_msg)
                agent_response = f"Error: Failed to get response. {e}" # Store simplified error in history
                logging.exception("Error occurred within the Streamlit chat input processing block.")

    st.session_state[message_history_key].append({"role": "assistant", "content": agent_response})
    st.rerun() # Explicitly rerun to update the chat display immediately

# --- Sidebar Information ---
st.sidebar.divider()
st.sidebar.header("Agent Details")
st.sidebar.caption(f"**Agent Name:** `{meta_agent.name}`")
st.sidebar.caption(f"**App Name:** `{APP_NAME}`")
st.sidebar.caption(f"**User ID:** `{USER_ID}`")
st.sidebar.caption(f"**ADK Session ID:** `{st.session_state.get(ADK_SESSION_STATE_KEY, 'N/A')}`")
