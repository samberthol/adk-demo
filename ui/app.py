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

# Add project root to Python path (Optional if PYTHONPATH is set in Dockerfile)
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
ADK_SESSION_STATE_KEY = f'adk_session_id_{APP_NAME}' # Use APP_NAME here
FASTAPI_INTERNAL_URL = "http://localhost:8000"

# Apply nest_asyncio
# ... (nest_asyncio logic as before) ...
try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully.")
except RuntimeError as e:
    if "cannot apply nest_asyncio" in str(e):
        logger.info("nest_asyncio already applied or cannot be applied again.")
    else:
        logger.error(f"Error applying nest_asyncio: {e}")

# --------------------------------------------------------------------------
# ADK Initialization Logic (Accepting Args)
# --------------------------------------------------------------------------
@st.cache_resource
def initialize_adk(app_name_arg: str, user_id_arg: str): # <--- Accept arguments
    """
    Initializes the ADK Runner and InMemorySessionService for the application.
    Manages the unique ADK session ID within the Streamlit session state.
    Uses passed arguments for app_name and user_id.
    """
    logger.info("--- ADK Init: Attempting to initialize Runner and Session Service... ---")
    session_service = InMemorySessionService()
    logger.info("--- ADK Init: InMemorySessionService instantiated. ---")

    # Use the arguments passed to the function
    runner = Runner(
        agent=meta_agent,
        app_name=app_name_arg,
        session_service=session_service
    )
    logger.info(f"--- ADK Init: Runner instantiated for agent '{meta_agent.name}'. ---")

    # Manage ADK session ID within Streamlit's session state
    # Use the globally defined key which depends on the constant APP_NAME
    if ADK_SESSION_STATE_KEY not in st.session_state:
        session_id = f"st_session_{app_name_arg}_{int(time.time())}_{os.urandom(4).hex()}" # Use arg here too if desired
        st.session_state[ADK_SESSION_STATE_KEY] = session_id
        logger.info(f"--- ADK Init: Generated new ADK session ID: {session_id} ---")
        try:
            # Create session using passed arguments
            session_service.create_session(
                app_name=app_name_arg,
                user_id=user_id_arg,
                session_id=session_id,
                state={}
            )
            logger.info("--- ADK Init: Successfully created new session in ADK SessionService.")
        except Exception as e:
            logger.exception(f"--- ADK Init: ERROR creating session {session_id}:")
            raise RuntimeError(f"Could not create initial ADK session {session_id}: {e}") from e
    else:
        session_id = st.session_state[ADK_SESSION_STATE_KEY]
        logger.info(f"--- ADK Init: Reusing existing ADK session ID from Streamlit state: {session_id} ---")
        try:
            # Check session using passed arguments
            existing_session = session_service.get_session(
                app_name=app_name_arg,
                user_id=user_id_arg,
                session_id=session_id
            )
            if not existing_session:
                logger.warning(f"--- ADK Init: Session {session_id} not found in memory. Recreating. State lost. ---")
                try:
                    # Recreate session using passed arguments
                    session_service.create_session(
                        app_name=app_name_arg,
                        user_id=user_id_arg,
                        session_id=session_id,
                        state={}
                    )
                    logger.info(f"--- ADK Init: Successfully recreated session {session_id}.")
                except Exception as e_recreate:
                    logger.exception(f"--- ADK Init: ERROR - Could not recreate missing session {session_id}:")
                    raise RuntimeError(f"Could not recreate missing ADK session {session_id}: {e_recreate}") from e_recreate
            else:
                 logger.info(f"--- ADK Init: Session {session_id} successfully found in SessionService memory.")
        except Exception as e_get:
             logger.error(f"--- ADK Init: Error trying to get session {session_id} from service: {e_get} ---")
             raise RuntimeError(f"Error checking ADK session {session_id} existence: {e_get}") from e_get

    logger.info(f"--- ADK Init: Initialization sequence complete. Runner is ready. Active Session ID: {session_id} ---")
    # Return runner and the determined session_id
    return runner, session_id

# --------------------------------------------------------------------------
# Async Runner Function (No changes needed here)
# --------------------------------------------------------------------------
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    # ... (async function remains the same) ...
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


# --------------------------------------------------------------------------
# Sync Wrapper (No changes needed here)
# --------------------------------------------------------------------------
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    # ... (sync wrapper remains the same) ...
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
    # Pass the global constants as arguments when calling the cached function
    adk_runner, current_adk_session_id = initialize_adk(APP_NAME, USER_ID) # <--- Pass args
    # Sidebar moved lower to allow checking query_params first
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize the ADK Runner or Session Service: {e}", icon="❌")
    st.error("Please check the terminal logs for more details and restart the application.")
    logger.exception("Critical ADK Initialization failed in Streamlit UI context.")
    st.stop() # Stop the app if ADK fails to initialize


# --- UI Rendering Logic ---
# ... (UI Rendering logic remains the same) ...
# Check query params FIRST to decide which UI to show
query_params = st.query_params.to_dict()
view_mode = query_params.get("view", ["streamlit"])[0] # Default to streamlit view

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
        # Clear the cache for the function. Since it now takes arguments,
        # simply calling clear() might work, or it might require specific args.
        # If clear() causes issues, you might need to remove this line or handle it differently.
        try:
            initialize_adk.clear()
            logger.info("Cleared ADK initialization cache.")
        except Exception as clear_ex:
            logger.warning(f"Could not clear initialize_adk cache: {clear_ex}")

        st.rerun()

    st.sidebar.divider()
    st.sidebar.header("Agent Details")
    # Use the cached runner instance to get agent name
    st.sidebar.caption(f"**Agent Name:** `{adk_runner.agent.name if adk_runner and adk_runner.agent else 'N/A'}`")
    st.sidebar.caption(f"**App Name:** `{APP_NAME}`") # Display global constant
    st.sidebar.caption(f"**User ID:** `{USER_ID}`") # Display global constant
    st.sidebar.caption(f"**Current Session ID:** `{st.session_state.get(ADK_SESSION_STATE_KEY, 'N/A')}`")

    # Add link to switch view
    st.sidebar.divider()
    st.sidebar.link_button("View FastAPI Test UI", st.get_option("server.baseUrlPath") + "?view=fastapi")