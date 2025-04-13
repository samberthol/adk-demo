# ui/app.py
import streamlit as st
import logging
import time
import os
import sys
from pathlib import Path
import asyncio
import nest_asyncio
import requests

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="GCP ADK Agent")

# (Optional) Add project root to Python path if PYTHONPATH isn't set/working
# project_root = str(Path(__file__).parent.parent)
# if project_root not in sys.path:
#     sys.path.append(project_root)

# Import ADK components AFTER set_page_config
try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
except ImportError as e:
    st.exception(f"Failed to import agent modules or ADK components: {e}")
    st.error("Ensure project structure and requirements are correct. App cannot start.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

# --- Constants ---
APP_NAME = "gcp_multi_agent_demo_streamlit"
USER_ID = f"st_user_{APP_NAME}" # Define a consistent User ID
ADK_SESSION_ID_KEY = f'adk_session_id_{APP_NAME}' # Key for session_id in st.session_state
ADK_SERVICE_KEY = f'adk_service_{APP_NAME}'      # Key for service in st.session_state
ADK_RUNNER_KEY = f'adk_runner_{APP_NAME}'       # Key for runner in st.session_state
FASTAPI_INTERNAL_URL = "http://localhost:8000"

# Apply nest_asyncio
try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully.")
except RuntimeError as e:
    # Ignore errors if already applied
    if "cannot apply nest_asyncio" not in str(e):
        logger.error(f"Error applying nest_asyncio: {e}")


# --------------------------------------------------------------------------
# ADK Initialization & Session Management within st.session_state
# --------------------------------------------------------------------------
def get_runner_and_session_id():
    """
    Initializes/retrieves Runner & Service using st.session_state.
    Gets/Creates session_id in st.session_state.
    Validates/Recreates session in the service on each call.

    Returns:
        tuple: (Runner instance, session_id string)
    """
    # Initialize Service and Runner in session_state if they don't exist
    if ADK_SERVICE_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new InMemorySessionService in st.session_state.")
        st.session_state[ADK_SERVICE_KEY] = InMemorySessionService()
    if ADK_RUNNER_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new Runner in st.session_state.")
        st.session_state[ADK_RUNNER_KEY] = Runner(
            agent=meta_agent,
            app_name=APP_NAME,
            session_service=st.session_state[ADK_SERVICE_KEY] # Use service from state
        )

    # Retrieve instances from session_state
    session_service = st.session_state[ADK_SERVICE_KEY]
    runner = st.session_state[ADK_RUNNER_KEY]

    # Get or create session ID in session_state
    if ADK_SESSION_ID_KEY not in st.session_state:
        session_id = f"st_session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_ID_KEY] = session_id
        logger.info(f"--- ADK Session Mgmt: Generated new ADK session ID: {session_id}")
        # Attempt initial creation in service
        try:
            session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={}
            )
            logger.info(f"--- ADK Session Mgmt: Created session {session_id} in service.")
        except Exception as e:
            logger.exception(f"--- ADK Session Mgmt: ERROR initially creating session {session_id}:")
            raise RuntimeError(f"Could not create initial ADK session {session_id}: {e}") from e
    else:
        session_id = st.session_state[ADK_SESSION_ID_KEY]
        logger.info(f"--- ADK Session Mgmt: Reusing ADK session ID from state: {session_id}")
        # Crucially, check and recreate session in service *every time* we reuse the ID
        try:
            existing = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            if not existing:
                logger.warning(f"--- ADK Session Mgmt: Session {session_id} not found in service. Recreating (state lost).")
                session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={}
                )
                logger.info(f"--- ADK Session Mgmt: Recreated session {session_id} in service.")
            # else: # Optional log
            #     logger.debug(f"--- ADK Session Mgmt: Session {session_id} confirmed exists in service.")
        except Exception as e:
            logger.exception(f"--- ADK Session Mgmt: Error checking/recreating session {session_id}:")
            # Raise error to prevent running with inconsistent state
            raise RuntimeError(f"Failed to validate/recreate ADK session {session_id}: {e}") from e

    return runner, session_id

# --------------------------------------------------------------------------
# Async Runner Function (remains the same)
# --------------------------------------------------------------------------
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    # ... (async function is unchanged) ...
    logger.info(f"\n--- ADK Run Async: Starting execution for session {session_id} ---")
    logger.info(f"--- ADK Run Async: Processing User Query (truncated): '{user_message_text[:150]}...' ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent encountered an issue and did not produce a final response]"
    start_time = time.time()
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                logger.info("--- ADK Run Async: Final response event received.")
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                else:
                    final_response_text = "[Agent finished but produced no text output]"
                    logger.warning(f"--- ADK Run Async: Final event received, but no text content found. Event: {event}")
                break
    except Exception as e:
        # Check for specific session errors if possible, otherwise log generic
        if "Session not found" in str(e):
             logger.error(f"--- ADK Run Async: Confirmed 'Session not found' error for {session_id} / {user_id}")
             final_response_text = "Error: Agent session expired or was lost. Please try clearing the session and starting again."
        else:
            logger.exception("--- ADK Run Async: !! EXCEPTION during agent execution:")
            final_response_text = f"Sorry, an error occurred: {e}"
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- ADK Run Async: Turn execution completed in {duration:.2f} seconds.")
    logger.info(f"--- ADK Run Async: Final Response (truncated): '{final_response_text[:150]}...' ---")
    return final_response_text


# --------------------------------------------------------------------------
# Sync Wrapper (remains the same)
# --------------------------------------------------------------------------
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    # ... (sync wrapper is unchanged) ...
    try:
        return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
    except RuntimeError as e:
        logger.exception("RuntimeError during asyncio.run in run_adk_sync:")
        return f"Error running agent task: {e}. Check logs."
    except Exception as e:
        logger.exception("Unexpected exception during run_adk_sync:")
        return f"An unexpected error occurred: {e}. Check logs."


# --- Initialize ADK Runner and Session for this run ---
try:
    # Call the function to get/init runner/service and validate/get session_id
    adk_runner, current_adk_session_id = get_runner_and_session_id()
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize ADK session: {e}", icon="❌")
    logger.exception("Critical ADK Initialization/Session Validation failed.")
    st.stop() # Stop the app if ADK setup fails

# --- UI Rendering Logic ---
# ... (UI Rendering logic remains the same, check view_mode first) ...
query_params = st.query_params.to_dict()
view_mode = query_params.get("view", ["streamlit"])[0]

if view_mode == "fastapi":
    # --- Display FastAPI HTML Page ---
    st.title("⚡ FastAPI Test UI (via Streamlit Proxy)")
    # ... (rest of FastAPI view) ...
    st.caption("Displaying content fetched from internal endpoint `/` on port 8000")
    st.markdown("---")
    try:
        response = requests.get(FASTAPI_INTERNAL_URL + "/", timeout=5)
        response.raise_for_status()
        st.markdown(response.text, unsafe_allow_html=True)
        st.markdown("---")
        st.caption(f"Status code from {FASTAPI_INTERNAL_URL}/: {response.status_code}")
        st.warning("""**Note:** This is a basic HTML proxy...""")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Could not fetch FastAPI UI from {FASTAPI_INTERNAL_URL}/: {req_err}")
        logger.error(f"Error fetching internal FastAPI UI: {req_err}")
    except Exception as ex:
        st.error(f"An unexpected error occurred while fetching FastAPI UI: {ex}")
        logger.exception("Unexpected error fetching/displaying FastAPI UI via proxy:")

else:
    # --- Display Streamlit Chat UI (Default) ---
    st.title("💬 GCP ADK Multi-Agent Demo")
    st.caption("🚀 Powered by Google ADK & Cloud Run")

    # Initialize chat history
    message_history_key = f"messages_{APP_NAME}"
    if message_history_key not in st.session_state:
        st.session_state[message_history_key] = [{"role": "assistant", "content": "Hello! How can I help you manage GCP resources or query data today?"}]

    # Display chat messages
    for message in st.session_state[message_history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("What would you like to do? (e.g., 'list vms', 'run query select * from `project.dataset.table` limit 5')"):
        if not current_adk_session_id:
             st.error("Agent session ID could not be established. Cannot process request.")
        # Add safety check for runner object
        elif not adk_runner:
             st.error("ADK Runner is not available. Cannot process request.")
        else:
            st.session_state[message_history_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                try:
                    # Pass the potentially validated/recreated runner and session_id
                    agent_response = run_adk_sync(adk_runner, current_adk_session_id, USER_ID, prompt)
                    message_placeholder.markdown(agent_response)
                    st.session_state[message_history_key].append({"role": "assistant", "content": agent_response})
                except Exception as e:
                    logger.exception("Error running ADK turn from Streamlit input:")
                    error_msg = f"An error occurred: {e}"
                    message_placeholder.markdown(error_msg)
                    st.session_state[message_history_key].append({"role": "assistant", "content": error_msg})

    # --- Sidebar ---
    # Add status indicator based on successful initialization
    if adk_runner and current_adk_session_id:
        st.sidebar.success(f"ADK Session Active\nID: ...{current_adk_session_id[-12:]}", icon="✅")
    else:
         st.sidebar.error("ADK Not Initialized", icon="❌")

    st.sidebar.divider()
    st.sidebar.title("Session Control")
    if st.sidebar.button("Clear Chat & Reset Session"): # Changed button label slightly
        logger.info(f"Clearing chat history and resetting session for user {USER_ID}, ADK session {current_adk_session_id}")
        st.session_state[message_history_key] = [] # Clear messages
        # Remove ADK related keys from session_state to force re-initialization
        if ADK_SESSION_ID_KEY in st.session_state:
            del st.session_state[ADK_SESSION_ID_KEY]
        if ADK_RUNNER_KEY in st.session_state:
            del st.session_state[ADK_RUNNER_KEY]
        if ADK_SERVICE_KEY in st.session_state:
            del st.session_state[ADK_SERVICE_KEY]
        logger.info("Cleared ADK keys from st.session_state.")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Agent Details")
    # Get runner name safely from session state if possible
    runner_instance = st.session_state.get(ADK_RUNNER_KEY)
    st.sidebar.caption(f"**Agent Name:** `{runner_instance.agent.name if runner_instance and runner_instance.agent else 'N/A'}`")
    st.sidebar.caption(f"**App Name:** `{APP_NAME}`")
    st.sidebar.caption(f"**User ID:** `{USER_ID}`")
    st.sidebar.caption(f"**Current Session ID:** `{st.session_state.get(ADK_SESSION_ID_KEY, 'N/A')}`")

    # Add link to switch view
    st.sidebar.divider()
    st.sidebar.link_button("View FastAPI Test UI", st.get_option("server.baseUrlPath") + "?view=fastapi")