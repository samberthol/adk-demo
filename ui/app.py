# ui/app.py
import streamlit as st
import logging
import time
import os
import sys
from pathlib import Path
import asyncio
import nest_asyncio # Import nest_asyncio

# Set page config as the first Streamlit command
st.set_page_config(layout="wide", page_title="GCP ADK Agent")

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import ADK components AFTER set_page_config
try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService # Keep using InMemorySessionService
    # Import Content and Part for constructing messages
    from google.genai.types import Content, Part
except ImportError as e:
    # Use st.exception to show the full traceback within the app if imports fail
    st.exception(f"Failed to import agent modules or ADK components: {e}")
    st.error("Ensure project structure and requirements are correct. App cannot start.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

# --- Constants ---
APP_NAME = "gcp_multi_agent_demo_streamlit" # Use a distinct name if running alongside FastAPI version
USER_ID = f"st_user_{APP_NAME}" # Define a consistent User ID for this Streamlit app instance
ADK_SESSION_STATE_KEY = f'adk_session_id_{APP_NAME}'

# Apply nest_asyncio needed for asyncio.run within Streamlit/other loops
try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully.")
except RuntimeError as e:
    if "cannot apply nest_asyncio" in str(e):
        logger.info("nest_asyncio already applied or cannot be applied again.")
    else:
        logger.error(f"Error applying nest_asyncio: {e}")

# --------------------------------------------------------------------------
# ADK Initialization Logic (Based on User's Working File)
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
    logger.info("--- ADK Init: Attempting to initialize Runner and Session Service... ---")
    session_service = InMemorySessionService()
    logger.info("--- ADK Init: InMemorySessionService instantiated. ---")

    runner = Runner(
        agent=meta_agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    logger.info(f"--- ADK Init: Runner instantiated for agent '{meta_agent.name}'. ---")

    # Manage ADK session ID within Streamlit's session state
    if ADK_SESSION_STATE_KEY not in st.session_state:
        session_id = f"st_session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_STATE_KEY] = session_id
        logger.info(f"--- ADK Init: Generated new ADK session ID: {session_id} ---")
        try:
            # Create the session record using arguments from the user's working file
            session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state={} # Use 'state' kwarg as in user's example
            )
            logger.info("--- ADK Init: Successfully created new session in ADK SessionService.")
        except Exception as e:
            # Log the specific error during creation
            logger.exception(f"--- ADK Init: ERROR creating session {session_id}:")
            # Raise a more informative error for Streamlit UI
            raise RuntimeError(f"Could not create initial ADK session {session_id}: {e}") from e
    else:
        session_id = st.session_state[ADK_SESSION_STATE_KEY]
        logger.info(f"--- ADK Init: Reusing existing ADK session ID from Streamlit state: {session_id} ---")
        try:
            # Check if the session actually exists in the service's memory
            # Using arguments from the user's working file
            existing_session = session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )

            if not existing_session:
                logger.warning(f"--- ADK Init: Session {session_id} not found in InMemorySessionService memory (likely due to script restart). Recreating session. State will be lost. ---")
                try:
                    # Recreate the session record in the service
                    session_service.create_session(
                        app_name=APP_NAME,
                        user_id=USER_ID,
                        session_id=session_id,
                        state={} # Use 'state' kwarg
                    )
                    logger.info(f"--- ADK Init: Successfully recreated session {session_id} in ADK SessionService.")
                except Exception as e_recreate:
                    logger.exception(f"--- ADK Init: ERROR - Could not recreate missing session {session_id}:")
                    # Raise error as state is inconsistent
                    raise RuntimeError(f"Could not recreate missing ADK session {session_id}: {e_recreate}") from e_recreate
            else:
                 logger.info(f"--- ADK Init: Session {session_id} successfully found in SessionService memory.")

        except Exception as e_get:
             # Handle potential errors during get_session itself
             logger.error(f"--- ADK Init: Error trying to get session {session_id} from service: {e_get} ---")
             # Decide how to proceed - raise error for clarity
             raise RuntimeError(f"Error checking ADK session {session_id} existence: {e_get}") from e_get


    logger.info(f"--- ADK Init: Initialization sequence complete. Runner is ready. Active Session ID: {session_id} ---")
    return runner, session_id

# --------------------------------------------------------------------------
# Async Runner Function (Matching User's Working File)
# --------------------------------------------------------------------------
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Asynchronously executes one turn of the ADK agent conversation.
    """
    logger.info(f"\n--- ADK Run Async: Starting execution for session {session_id} ---")
    logger.info(f"--- ADK Run Async: Processing User Query (truncated): '{user_message_text[:150]}...' ---")
    content = Content(
        role='user',
        parts=[Part(text=user_message_text)]
    )
    final_response_text = "[Agent encountered an issue and did not produce a final response]"
    start_time = time.time()
    try:
        # This uses runner.run_async as in the user's working file structure
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                logger.info("--- ADK Run Async: Final response event received.")
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                else:
                    final_response_text = "[Agent finished but produced no text output]"
                    logger.warning(f"--- ADK Run Async: Final event received, but no text content found. Event: {event}")
                break
            # else: # Optional: Log intermediate events if needed for debugging
            #     logger.debug(f"--- ADK Run Async: Intermediate event: {event.type}")

    except Exception as e:
        logger.exception("--- ADK Run Async: !! EXCEPTION during agent execution:")
        final_response_text = f"Sorry, an error occurred: {e}" # Simpler error for UI

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"--- ADK Run Async: Turn execution completed in {duration:.2f} seconds.")
    logger.info(f"--- ADK Run Async: Final Response (truncated): '{final_response_text[:150]}...' ---")
    return final_response_text

# --------------------------------------------------------------------------
# Sync Wrapper for Streamlit (Matching User's Working File)
# --------------------------------------------------------------------------
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Synchronous wrapper that executes the asynchronous run_adk_async function
    using asyncio.run(). Requires nest_asyncio.apply() to have been called.
    """
    try:
        # Use asyncio.run as in the user's working file structure
        return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
    except RuntimeError as e:
        # Handle potential asyncio event loop issues within Streamlit more gracefully
        logger.exception("RuntimeError during asyncio.run in run_adk_sync:")
        # Return the error message to be displayed in the UI
        return f"Error running agent task: {e}. Check logs."
    except Exception as e:
        logger.exception("Unexpected exception during run_adk_sync:")
        return f"An unexpected error occurred: {e}. Check logs."


# --- Streamlit UI ---
st.title("💬 GCP ADK Multi-Agent Demo")
st.caption("🚀 Powered by Google ADK & Cloud Run")

# --- Initialize ADK Runner and Session ---
try:
    # Call the initialization function based on user's working file
    adk_runner, current_adk_session_id = initialize_adk()
    st.sidebar.success(f"ADK Initialized\nSession: ...{current_adk_session_id[-12:]}", icon="✅")
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize the ADK Runner or Session Service: {e}", icon="❌")
    st.error("Please check the terminal logs for more details and restart the application.")
    logger.exception("Critical ADK Initialization failed in Streamlit UI context.")
    st.stop() # Stop the app if ADK fails to initialize

# --- Chat Interface Implementation ---
message_history_key = f"messages_{APP_NAME}"
if message_history_key not in st.session_state:
    st.session_state[message_history_key] = [{"role": "assistant", "content": "Hello! How can I help you manage GCP resources or query data today?"}]

# Display chat messages from history
for message in st.session_state[message_history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) # Keep unsafe_allow_html=False for safety

# Handle user input
if prompt := st.chat_input("Enter your request (e.g., 'Create a VM named my-vm')"):
    st.session_state[message_history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...") # Show thinking message immediately
        try:
            # Call the synchronous wrapper which runs the async function
            agent_response = run_adk_sync(adk_runner, current_adk_session_id, USER_ID, prompt)
            message_placeholder.markdown(agent_response) # Update with actual response
        except Exception as e:
            # Catch errors from run_adk_sync itself (less likely now with internal handling)
            error_msg = f"Sorry, an error occurred: {e}"
            st.error(error_msg)
            agent_response = error_msg # Store error in history
            logger.exception("Error occurred within the Streamlit chat input processing block (calling run_adk_sync).")

    st.session_state[message_history_key].append({"role": "assistant", "content": agent_response})
    # Rerun might not be strictly necessary if using placeholders correctly, but can ensure state consistency
    # st.rerun()


# --- Sidebar Controls and Information ---
st.sidebar.divider()
st.sidebar.title("Session Control")
if st.sidebar.button("Clear Chat & Reset Session ID"):
    logger.info(f"Clearing chat history for user {USER_ID}, ADK session {current_adk_session_id}")
    st.session_state[message_history_key] = [] # Clear messages
    # Remove the session ID key from Streamlit state, forcing regeneration on next run
    if ADK_SESSION_STATE_KEY in st.session_state:
        del st.session_state[ADK_SESSION_STATE_KEY]
        logger.info(f"Removed ADK session key '{ADK_SESSION_STATE_KEY}' from Streamlit state.")
    # Clear the ADK runner/service cache as well
    initialize_adk.clear()
    logger.info("Cleared ADK initialization cache.")
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Agent Details")
st.sidebar.caption(f"**Agent Name:** `{adk_runner.agent.name if adk_runner and adk_runner.agent else 'N/A'}`")
st.sidebar.caption(f"**App Name:** `{APP_NAME}`")
st.sidebar.caption(f"**User ID:** `{USER_ID}`")
st.sidebar.caption(f"**Current Session ID:** `{st.session_state.get(ADK_SESSION_STATE_KEY, 'N/A')}`")