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
APP_NAME = "gcp_multi_agent_demo_streamlit"
USER_ID_PREFIX = "st_user_"
ADK_SESSION_PREFIX = f'adk_session_{APP_NAME}_'

# --- ADK Initialization ---
# Use Streamlit session state to store ADK components to persist across reruns
if 'session_service' not in st.session_state:
    st.session_state.session_service = InMemorySessionService()
    logger.info("--- Streamlit ADK Init: InMemorySessionService instantiated. ---")

if 'adk_runner' not in st.session_state:
    st.session_state.adk_runner = Runner(
        agent=meta_agent,
        app_name=APP_NAME,
        session_service=st.session_state.session_service
    )
    logger.info(f"--- Streamlit ADK Init: Runner instantiated for agent '{meta_agent.name}'. ---")

# --- Helper Functions for ADK Interaction (Using Synchronous Methods) ---

# Apply nest_asyncio once if needed, though not strictly required for sync calls
# Might help if other parts of streamlit or libraries use asyncio internally
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


def get_or_create_adk_session_sync(user_id: str) -> str:
    """Gets existing or creates a new ADK session ID for a Streamlit user (Sync)."""
    session_key = f"adk_session_id_{user_id}"
    service = st.session_state.session_service # Reference from state

    if session_key in st.session_state:
        session_id = st.session_state[session_key]
        logger.info(f"--- Streamlit ADK Sync: Reusing existing ADK session for {user_id}: {session_id} ---")
        try:
            # Use synchronous ADK methods
            existing = service.get_session(APP_NAME, user_id, session_id)
            if not existing:
                logger.warning(f"--- Streamlit ADK Sync: Session {session_id} for {user_id} not found in service. Recreating.")
                service.create_session(APP_NAME, user_id, session_id, {})
        except Exception as e:
             logger.error(f"--- Streamlit ADK Sync: Error checking/recreating session {session_id} for {user_id}: {e}")
             if session_key in st.session_state: del st.session_state[session_key]
             return get_or_create_adk_session_sync(user_id) # Recurse
        return session_id
    else:
        session_id = f"{ADK_SESSION_PREFIX}{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[session_key] = session_id
        logger.info(f"--- Streamlit ADK Sync: Generated new ADK session for {user_id}: {session_id} ---")
        try:
            service.create_session(APP_NAME, user_id, session_id, {})
            logger.info(f"--- Streamlit ADK Sync: Successfully created new session {session_id} in SessionService.")
        except Exception as e:
            logger.exception(f"--- Streamlit ADK Sync: FATAL ERROR creating session {session_id} for {user_id}:")
            if session_key in st.session_state: del st.session_state[session_key]
            raise
        return session_id


def run_adk_turn_sync(user_id: str, session_id: str, user_message_text: str) -> str:
    """Runs a single turn with the ADK agent synchronously."""
    logger.info(f"--- Streamlit ADK Run Sync: Session {session_id}, User: {user_id}, Query: '{user_message_text[:100]}...' ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"
    adk_runner = st.session_state.adk_runner # Reference from state

    try:
        # Use the synchronous runner method
        for event in adk_runner.run(user_id=user_id, session_id=session_id, new_message=content):
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
if "messages" not in st.session_state:
    st.session_state.messages = []

# Generate a unique user ID for this Streamlit session if needed
if "user_id" not in st.session_state:
    st.session_state.user_id = f"{USER_ID_PREFIX}{os.urandom(4).hex()}"
    logger.info(f"Generated new Streamlit user ID: {st.session_state.user_id}")

# Get or create the ADK session ID for this user
try:
     # Call the synchronous version of the function
     adk_session_id = get_or_create_adk_session_sync(st.session_state.user_id)
except Exception as e:
     # Display error and stop if session initialization fails critically
     st.error(f"Fatal error initializing agent session: {e}. Please refresh.")
     st.stop()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to do? (e.g., 'list vms', 'run query select * from `project.dataset.table` limit 5')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
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
            agent_response = run_adk_turn_sync(st.session_state.user_id, adk_session_id, prompt)
            message_placeholder.markdown(agent_response) # Update placeholder with actual response
            st.session_state.messages.append({"role": "assistant", "content": agent_response}) # Add response to history

        except Exception as e:
            logger.exception("Error running ADK turn from Streamlit input:")
            error_msg = f"An error occurred: {e}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add a button to clear the chat history and session
st.sidebar.title("Session Control")
if st.sidebar.button("Clear Chat History"):
    logger.info(f"Clearing chat history for user {st.session_state.user_id}, ADK session {adk_session_id}")
    st.session_state.messages = []
    # Optional: Consider if you want to explicitly delete/reset the ADK session
    # on the server side here as well. For InMemorySessionService, restarting the
    # container effectively clears it, but explicit deletion might be cleaner.
    # Example (requires implementing delete_session in service if available):
    # try:
    #     st.session_state.session_service.delete_session(APP_NAME, st.session_state.user_id, adk_session_id)
    # except Exception as del_e:
    #     logger.error(f"Error trying to delete ADK session {adk_session_id}: {del_e}")
    st.rerun()

if st.sidebar.button("Clear Full Session State (Debug)"):
    logger.warning("Clearing full Streamlit session state.")
    keys_to_keep = ['query_params'] # Example, adjust if needed
    kept_values = {k: st.session_state[k] for k in keys_to_keep if k in st.session_state}

    st.session_state.clear() # Clear everything

    # Restore kept values
    for k, v in kept_values.items():
        st.session_state[k] = v
    st.rerun()