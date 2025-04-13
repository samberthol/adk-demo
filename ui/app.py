# ui/app.py
import streamlit as st
import logging
import time
import os
import sys
from pathlib import Path
import asyncio
import nest_asyncio
import requests # <--- Import requests

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="GCP ADK Agent")

# Add project root to Python path
# ... (rest of imports and path setup as before) ...
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
# ... (logging config as before) ...
logger = logging.getLogger("streamlit_app")

# --- Constants ---
# ... (constants as before) ...
FASTAPI_INTERNAL_URL = "http://localhost:8000" # Internal address for FastAPI

# Apply nest_asyncio
# ... (nest_asyncio logic as before) ...

# --- ADK Initialization ---
# ... (initialize_adk function using @st.cache_resource as before) ...
@st.cache_resource
def initialize_adk():
    # ... (exact code from previous correct version) ...
    logger.info("--- ADK Init: Attempting to initialize Runner and Session Service... ---")
    session_service = InMemorySessionService()
    logger.info("--- ADK Init: InMemorySessionService instantiated. ---")
    runner = Runner(
        agent=meta_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    logger.info(f"--- ADK Init: Runner instantiated for agent '{meta_agent.name}'. ---")
    if ADK_SESSION_STATE_KEY not in st.session_state:
        session_id = f"st_session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_STATE_KEY] = session_id
        logger.info(f"--- ADK Init: Generated new ADK session ID: {session_id} ---")
        try:
            session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={}
            )
            logger.info("--- ADK Init: Successfully created new session in ADK SessionService.")
        except Exception as e:
            logger.exception(f"--- ADK Init: ERROR creating session {session_id}:")
            raise RuntimeError(f"Could not create initial ADK session {session_id}: {e}") from e
    else:
        session_id = st.session_state[ADK_SESSION_STATE_KEY]
        logger.info(f"--- ADK Init: Reusing existing ADK session ID from Streamlit state: {session_id} ---")
        try:
            existing_session = session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )
            if not existing_session:
                logger.warning(f"--- ADK Init: Session {session_id} not found in InMemorySessionService memory. Recreating session. State will be lost. ---")
                try:
                    session_service.create_session(
                        app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={}
                    )
                    logger.info(f"--- ADK Init: Successfully recreated session {session_id} in ADK SessionService.")
                except Exception as e_recreate:
                    logger.exception(f"--- ADK Init: ERROR - Could not recreate missing session {session_id}:")
                    raise RuntimeError(f"Could not recreate missing ADK session {session_id}: {e_recreate}") from e_recreate
            else:
                 logger.info(f"--- ADK Init: Session {session_id} successfully found in SessionService memory.")
        except Exception as e_get:
             logger.error(f"--- ADK Init: Error trying to get session {session_id} from service: {e_get} ---")
             raise RuntimeError(f"Error checking ADK session {session_id} existence: {e_get}") from e_get
    logger.info(f"--- ADK Init: Initialization sequence complete. Runner is ready. Active Session ID: {session_id} ---")
    return runner, session_id


# --- Async Runner Function ---
# ... (run_adk_async function as before) ...
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
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
        logger.exception("--- ADK Run Async: !! EXCEPTION during agent execution:")
        final_response_text = f"Sorry, an error occurred: {e}"
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- ADK Run Async: Turn execution completed in {duration:.2f} seconds.")
    logger.info(f"--- ADK Run Async: Final Response (truncated): '{final_response_text[:150]}...' ---")
    return final_response_text

# --- Sync Wrapper ---
# ... (run_adk_sync function as before) ...
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    try:
        return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
    except RuntimeError as e:
        logger.exception("RuntimeError during asyncio.run in run_adk_sync:")
        return f"Error running agent task: {e}. Check logs."
    except Exception as e:
        logger.exception("Unexpected exception during run_adk_sync:")
        return f"An unexpected error occurred: {e}. Check logs."


# --- Initialize ADK Runner and Session ---
try:
    adk_runner, current_adk_session_id = initialize_adk()
    # Sidebar moved lower to allow checking query_params first
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize the ADK Runner or Session Service: {e}", icon="❌")
    st.error("Please check the terminal logs for more details and restart the application.")
    logger.exception("Critical ADK Initialization failed in Streamlit UI context.")
    st.stop() # Stop the app if ADK fails to initialize


# --- UI Rendering Logic ---

# Check query params FIRST to decide which UI to show
query_params = st.query_params.to_dict()
view_mode = query_params.get("view", ["streamlit"])[0] # Default to streamlit view

if view_mode == "fastapi":
    # --- Display FastAPI HTML Page ---
    st.title("⚡ FastAPI Test UI (via Streamlit Proxy)")
    st.caption("Displaying content fetched from internal endpoint `/` on port 8000")
    st.markdown("---")
    try:
        # Make request to the internal FastAPI endpoint
        response = requests.get(FASTAPI_INTERNAL_URL + "/", timeout=5) # Added timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # Display the fetched HTML content
        # WARNING: unsafe_allow_html=True can be a security risk if the content isn't trusted.
        st.markdown(response.text, unsafe_allow_html=True)
        st.markdown("---")
        st.caption(f"Status code from {FASTAPI_INTERNAL_URL}/: {response.status_code}")

        st.warning("""
            **Note:** This is a basic HTML proxy.
            * JavaScript within the FastAPI page might not function correctly, especially WebSocket connections which need specific handling.
            * Relative links for CSS/JS might be broken.
            * This is a workaround and not a standard way to integrate FastAPI UIs.
            """)

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
        else:
            st.session_state[message_history_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                try:
                    agent_response = run_adk_sync(adk_runner, USER_ID, current_adk_session_id, prompt)
                    message_placeholder.markdown(agent_response)
                    st.session_state[message_history_key].append({"role": "assistant", "content": agent_response})
                except Exception as e:
                    logger.exception("Error running ADK turn from Streamlit input:")
                    error_msg = f"An error occurred: {e}"
                    message_placeholder.markdown(error_msg)
                    st.session_state[message_history_key].append({"role": "assistant", "content": error_msg})

    # --- Sidebar ---
    st.sidebar.success(f"ADK Session Active\nID: ...{current_adk_session_id[-12:]}", icon="✅")
    st.sidebar.divider()
    st.sidebar.title("Session Control")
    if st.sidebar.button("Clear Chat & Reset Session ID"):
        logger.info(f"Clearing chat history for user {USER_ID}, ADK session {current_adk_session_id}")
        st.session_state[message_history_key] = []
        if ADK_SESSION_STATE_KEY in st.session_state:
            del st.session_state[ADK_SESSION_STATE_KEY]
            logger.info(f"Removed ADK session key '{ADK_SESSION_STATE_KEY}' from Streamlit state.")
        initialize_adk.clear()
        logger.info("Cleared ADK initialization cache.")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Agent Details")
    st.sidebar.caption(f"**Agent Name:** `{adk_runner.agent.name if adk_runner and adk_runner.agent else 'N/A'}`")
    st.sidebar.caption(f"**App Name:** `{APP_NAME}`")
    st.sidebar.caption(f"**User ID:** `{USER_ID}`")
    st.sidebar.caption(f"**Current Session ID:** `{st.session_state.get(ADK_SESSION_STATE_KEY, 'N/A')}`")

    # Add link to switch view
    st.sidebar.divider()
    st.sidebar.link_button("View FastAPI Test UI", st.get_option("server.baseUrlPath") + "?view=fastapi")