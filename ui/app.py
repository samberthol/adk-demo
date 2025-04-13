# ui/app.py
import streamlit as st
import logging
import time
import os
import sys
from pathlib import Path
import asyncio
import nest_asyncio
# Removed: import requests

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="GCP ADK Agent Chat") # Updated title

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
# Removed: FASTAPI_INTERNAL_URL

# Apply nest_asyncio
try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully.")
except RuntimeError as e:
    if "cannot apply nest_asyncio" not in str(e): logger.error(f"Error applying nest_asyncio: {e}")

# --------------------------------------------------------------------------
# ADK Initialization & Session Management within st.session_state
# --------------------------------------------------------------------------
def get_runner_and_session_id():
    # ... (This function remains exactly the same as the last working version) ...
    if ADK_SERVICE_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new InMemorySessionService in st.session_state.")
        st.session_state[ADK_SERVICE_KEY] = InMemorySessionService()
    if ADK_RUNNER_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new Runner in st.session_state.")
        st.session_state[ADK_RUNNER_KEY] = Runner(
            agent=meta_agent,
            app_name=APP_NAME,
            session_service=st.session_state[ADK_SERVICE_KEY]
        )
    session_service = st.session_state[ADK_SERVICE_KEY]
    runner = st.session_state[ADK_RUNNER_KEY]
    if ADK_SESSION_ID_KEY not in st.session_state:
        session_id = f"st_session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_ID_KEY] = session_id
        logger.info(f"--- ADK Session Mgmt: Generated new ADK session ID: {session_id}")
        try:
            session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={})
            logger.info(f"--- ADK Session Mgmt: Created session {session_id} in service.")
        except Exception as e:
            logger.exception(f"--- ADK Session Mgmt: ERROR initially creating session {session_id}:")
            raise RuntimeError(f"Could not create initial ADK session {session_id}: {e}") from e
    else:
        session_id = st.session_state[ADK_SESSION_ID_KEY]
        logger.info(f"--- ADK Session Mgmt: Reusing ADK session ID from state: {session_id}")
        try:
            existing = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            if not existing:
                logger.warning(f"--- ADK Session Mgmt: Session {session_id} not found in service. Recreating (state lost).")
                session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={})
                logger.info(f"--- ADK Session Mgmt: Recreated session {session_id} in service.")
        except Exception as e:
            logger.exception(f"--- ADK Session Mgmt: Error checking/recreating session {session_id}:")
            raise RuntimeError(f"Failed to validate/recreate ADK session {session_id}: {e}") from e
    return runner, session_id

# --------------------------------------------------------------------------
# Async Runner Function (remains the same)
# --------------------------------------------------------------------------
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    # ... (async function remains exactly the same) ...
    logger.info(f"\n--- ADK Run Async: Starting execution for session {session_id} ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"
    start_time = time.time()
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                else:
                    final_response_text = "[Agent finished with no text output]"
                break
    except Exception as e:
        if "Session not found" in str(e):
             logger.error(f"--- ADK Run Async: Confirmed 'Session not found' error for {session_id} / {user_id}")
             final_response_text = "Error: Agent session expired or was lost. Please try clearing the session and starting again."
        else:
            logger.exception("--- ADK Run Async: !! EXCEPTION during agent execution:")
            final_response_text = f"Sorry, an error occurred: {e}"
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- ADK Run Async: Turn execution completed in {duration:.2f} seconds.")
    return final_response_text

# --------------------------------------------------------------------------
# Sync Wrapper (remains the same)
# --------------------------------------------------------------------------
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    # ... (sync wrapper remains exactly the same) ...
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
    adk_runner, current_adk_session_id = get_runner_and_session_id()
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize ADK session: {e}", icon="❌")
    logger.exception("Critical ADK Initialization/Session Validation failed.")
    st.stop()

# --- UI Rendering Logic ---
# Removed the view_mode check, always show Streamlit UI

# --- Display Streamlit Chat UI (Default) ---
st.title("💬 GCP ADK Multi-Agent Chat") # Updated title
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
                agent_response = run_adk_sync(adk_runner, current_adk_session_id, USER_ID, prompt)
                message_placeholder.markdown(agent_response)
                st.session_state[message_history_key].append({"role": "assistant", "content": agent_response})
            except Exception as e:
                logger.exception("Error running ADK turn from Streamlit input:")
                error_msg = f"An error occurred: {e}"
                message_placeholder.markdown(error_msg)
                st.session_state[message_history_key].append({"role": "assistant", "content": error_msg})

# --- Sidebar ---
if adk_runner and current_adk_session_id:
    st.sidebar.success(f"ADK Session Active\nID: ...{current_adk_session_id[-12:]}", icon="✅")
else:
     st.sidebar.error("ADK Not Initialized", icon="❌")

st.sidebar.divider()
st.sidebar.title("Session Control")
if st.sidebar.button("Clear Chat & Reset Session"):
    logger.info(f"Clearing chat history and resetting session for user {USER_ID}, ADK session {current_adk_session_id}")
    st.session_state[message_history_key] = []
    if ADK_SESSION_ID_KEY in st.session_state: del st.session_state[ADK_SESSION_ID_KEY]
    if ADK_RUNNER_KEY in st.session_state: del st.session_state[ADK_RUNNER_KEY]
    if ADK_SERVICE_KEY in st.session_state: del st.session_state[ADK_SERVICE_KEY]
    logger.info("Cleared ADK keys from st.session_state.")
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Agent Details")
runner_instance = st.session_state.get(ADK_RUNNER_KEY)
st.sidebar.caption(f"**Agent Name:** `{runner_instance.agent.name if runner_instance and runner_instance.agent else 'N/A'}`")
st.sidebar.caption(f"**App Name:** `{APP_NAME}`")
st.sidebar.caption(f"**User ID:** `{USER_ID}`")
st.sidebar.caption(f"**Current Session ID:** `{st.session_state.get(ADK_SESSION_ID_KEY, 'N/A')}`")

# Removed: Link button to FastAPI view