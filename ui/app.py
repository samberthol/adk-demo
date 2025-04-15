# ui/app.py
import streamlit as st
import logging
import time
import os
import sys
from pathlib import Path
import asyncio
import nest_asyncio

# --- Constants ---
APP_NAME = "gcp_multi_agent_demo_streamlit"
USER_ID = f"st_user_{APP_NAME}" # Define a consistent User ID
ADK_SESSION_ID_KEY = f'adk_session_id_{APP_NAME}' # Key for session_id in st.session_state
ADK_SERVICE_KEY = f'adk_service_{APP_NAME}'      # Key for service in st.session_state
ADK_RUNNER_KEY = f'adk_runner_{APP_NAME}'       # Key for runner in st.session_state
MESSAGE_HISTORY_KEY = f"messages_{APP_NAME}"     # Key for chat history
GCP_LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Google_Cloud_Platform_logo.svg/320px-Google_Cloud_Platform_logo.svg.png" # Google Cloud Logo URL

# --- Page Configuration (Apply First) ---
st.set_page_config(
    layout="wide",
    page_title="GCP Agent Hub",
    page_icon="☁️" # Set a page icon
    # theme="dark" # Streamlit now defaults to system theme, user can toggle. Explicitly set if desired.
    )

# --- ADK Imports (After Page Config) ---
# Add project root to Python path if needed (adjust path as necessary)
# project_root = str(Path(__file__).parent.parent)
# if project_root not in sys.path:
#     sys.path.append(project_root)
#     logger.info(f"Added {project_root} to sys.path")

try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
except ImportError as e:
    st.exception(f"Failed to import agent modules or ADK components: {e}")
    st.error("Ensure project structure and requirements are correct. App cannot start.")
    st.stop()

# --- Logging Configuration ---
# Reconfigure logging to ensure level is set correctly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("streamlit_app")
logger.info("Streamlit App Logger Initialized.")


# --- Apply nest_asyncio ---
# Needed for running async ADK code within Streamlit's sync environment
try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully.")
except RuntimeError as e:
    # Ignore error if loop is already running (common in Streamlit reruns)
    if "cannot apply nest_asyncio" not in str(e):
         logger.error(f"Error applying nest_asyncio: {e}")


# --------------------------------------------------------------------------
# ADK Initialization & Session Management within st.session_state
# --------------------------------------------------------------------------
def get_runner_and_session_id():
    """
    Initializes or retrieves the ADK Runner and Session ID using Streamlit's session state.
    Handles creation and validation of the ADK session.
    """
    # Initialize Session Service if not already in state
    if ADK_SERVICE_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new InMemorySessionService in st.session_state.")
        st.session_state[ADK_SERVICE_KEY] = InMemorySessionService()

    # Initialize Runner if not already in state
    if ADK_RUNNER_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new Runner in st.session_state.")
        st.session_state[ADK_RUNNER_KEY] = Runner(
            agent=meta_agent,
            app_name=APP_NAME,
            session_service=st.session_state[ADK_SERVICE_KEY]
        )

    session_service = st.session_state[ADK_SERVICE_KEY]
    runner = st.session_state[ADK_RUNNER_KEY]

    # Get or create ADK Session ID
    if ADK_SESSION_ID_KEY not in st.session_state:
        session_id = f"st_session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_ID_KEY] = session_id
        logger.info(f"--- ADK Session Mgmt: Generated new ADK session ID: {session_id}")
        try:
            # Create the session in the ADK service
            session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={})
            logger.info(f"--- ADK Session Mgmt: Created session {session_id} in service.")
        except Exception as e:
            logger.exception(f"--- ADK Session Mgmt: ERROR initially creating session {session_id}:")
            # Clean up potentially bad state keys if creation fails
            if ADK_SESSION_ID_KEY in st.session_state: del st.session_state[ADK_SESSION_ID_KEY]
            raise RuntimeError(f"Could not create initial ADK session {session_id}: {e}") from e
    else:
        # Reuse existing session ID from state
        session_id = st.session_state[ADK_SESSION_ID_KEY]
        logger.info(f"--- ADK Session Mgmt: Reusing ADK session ID from state: {session_id}")
        try:
            # Validate if session still exists in the service backend
            existing = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            if not existing:
                # If session is lost in backend (e.g., service restart), recreate it
                logger.warning(f"--- ADK Session Mgmt: Session {session_id} not found in service. Recreating (state lost).")
                session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={})
                logger.info(f"--- ADK Session Mgmt: Recreated session {session_id} in service.")
        except Exception as e:
            logger.exception(f"--- ADK Session Mgmt: Error checking/recreating session {session_id}:")
            # Clean up potentially bad state keys if validation fails
            if ADK_SESSION_ID_KEY in st.session_state: del st.session_state[ADK_SESSION_ID_KEY]
            raise RuntimeError(f"Failed to validate/recreate ADK session {session_id}: {e}") from e

    return runner, session_id

# --------------------------------------------------------------------------
# Async Runner Function
# --------------------------------------------------------------------------
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Runs a turn of the ADK agent asynchronously.
    """
    logger.info(f"\n--- ADK Run Async: Starting execution for session {session_id} ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]" # Default response
    start_time = time.time()

    try:
        # Iterate through agent events asynchronously
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            # Check for the final response event
            if event.is_final_response():
                # Extract text from the final response content parts
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                else:
                    final_response_text = "[Agent finished with no text output]"
                break # Exit loop once final response is received
    except Exception as e:
        # Handle potential session not found errors specifically
        if "Session not found" in str(e):
             logger.error(f"--- ADK Run Async: Confirmed 'Session not found' error for {session_id} / {user_id}")
             final_response_text = "Error: Agent session expired or was lost. Please try clearing the session and starting again."
        else:
            # Log other exceptions
            logger.exception("--- ADK Run Async: !! EXCEPTION during agent execution:")
            final_response_text = f"Sorry, an error occurred during agent execution: {e}"
    finally:
        # Log execution duration
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- ADK Run Async: Turn execution completed in {duration:.2f} seconds.")

    return final_response_text

# --------------------------------------------------------------------------
# Sync Wrapper for running async code in Streamlit
# --------------------------------------------------------------------------
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Synchronous wrapper to call the async ADK runner function using asyncio.run.
    Handles potential runtime errors during async execution.
    """
    try:
        # Run the async function using the current event loop or create a new one
        return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
    except RuntimeError as e:
        # Handle cases where asyncio.run might conflict with existing loops (less common with nest_asyncio)
        logger.exception("RuntimeError during asyncio.run in run_adk_sync:")
        return f"Error running agent task: {e}. Check logs."
    except Exception as e:
        # Catch any other unexpected errors during the sync call
        logger.exception("Unexpected exception during run_adk_sync:")
        return f"An unexpected error occurred: {e}. Check logs."

# --- Initialize ADK Runner and Session for this run ---
# Encapsulate initialization in a try-except block to handle potential errors gracefully
try:
    adk_runner, current_adk_session_id = get_runner_and_session_id()
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize ADK session: {e}", icon="❌")
    logger.exception("Critical ADK Initialization/Session Validation failed.")
    # Stop the app if ADK cannot be initialized
    st.stop()


# --- Sidebar UI ---
with st.sidebar:
    # Add Google Cloud logo at the top of the sidebar
    st.image(GCP_LOGO_URL, width=200) # Adjust width as needed
    st.divider() # Add a visual divider

    st.header("⚙️ Session Info") # Use header for better structure
    # Display session status with icons
    if current_adk_session_id:
        st.success(f"Session Active ✅")
    else:
         st.error("Session Inactive ❌")

    # Session reset button
    if st.button("🔄 Clear Chat & Reset Session"):
        logger.info(f"Clearing chat history and resetting session for user {USER_ID}, ADK session {current_adk_session_id}")
        # Clear relevant keys from Streamlit's session state
        st.session_state.pop(MESSAGE_HISTORY_KEY, None)
        st.session_state.pop(ADK_SESSION_ID_KEY, None)
        st.session_state.pop(ADK_RUNNER_KEY, None) # Remove runner to force re-init
        st.session_state.pop(ADK_SERVICE_KEY, None) # Remove service to force re-init
        logger.info("Cleared ADK keys from st.session_state.")
        st.toast("Session Cleared!") # Provide user feedback
        st.rerun() # Rerun the app to reflect changes

    st.divider() # Add another divider

    st.header("🤖 Agent Details") # Use header
    # Retrieve runner instance safely from state
    runner_instance = st.session_state.get(ADK_RUNNER_KEY)
    # Display agent details with markdown for potential styling/icons
    st.markdown(f"**Agent:** `{runner_instance.agent.name if runner_instance and runner_instance.agent else 'N/A'}`")
    st.markdown(f"**App:** `{APP_NAME}`")
    st.markdown(f"**User:** `{USER_ID}`")

    # Put full session ID in an expander to keep sidebar tidy
    with st.expander("Show Full Session ID"):
        st.code(st.session_state.get(ADK_SESSION_ID_KEY, 'N/A'))

# --- Main Chat Interface UI ---
st.title("☁️ GCP Agent Hub") # Updated title with icon
st.caption("Powered by Google ADK")

# Initialize chat history in session state if it doesn't exist
if MESSAGE_HISTORY_KEY not in st.session_state:
    # Start with a welcoming message from the assistant
    st.session_state[MESSAGE_HISTORY_KEY] = [{"role": "assistant", "content": "Hello Samuel! As a fellow Googler passionate about AI/ML and Cloud, how can I assist you today?"}]

# Display existing chat messages from history
for message in st.session_state[MESSAGE_HISTORY_KEY]:
    # Assign different avatars based on the role
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"]) # Render message content as Markdown

# Handle new user input via chat input widget
if prompt := st.chat_input("Ask about GCP resources, data, or GitHub..."):
    # Ensure ADK runner and session ID are available before processing
    if not current_adk_session_id:
         st.error("Agent session ID could not be established. Cannot process request.", icon="❌")
    elif not adk_runner:
         st.error("ADK Runner is not available. Cannot process request.", icon="❌")
    else:
        # Add user message to chat history and display it
        st.session_state[MESSAGE_HISTORY_KEY].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Display assistant response placeholder and spinner while processing
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty() # Create placeholder for dynamic update
            with st.spinner("Agent is processing..."): # Show spinner
                try:
                    # Call the ADK agent (sync wrapper around async call)
                    agent_response = run_adk_sync(adk_runner, current_adk_session_id, USER_ID, prompt)
                    # Display the actual agent response
                    message_placeholder.markdown(agent_response)
                    # Add assistant response to chat history
                    st.session_state[MESSAGE_HISTORY_KEY].append({"role": "assistant", "content": agent_response})
                except Exception as e:
                    # Handle errors during agent execution
                    logger.exception("Error running ADK turn from Streamlit input:")
                    error_msg = f"An error occurred: {e}"
                    # Display error message in the chat interface
                    message_placeholder.error(error_msg, icon="🚨")
                    # Add error message to chat history for context
                    st.session_state[MESSAGE_HISTORY_KEY].append({"role": "assistant", "content": error_msg})
