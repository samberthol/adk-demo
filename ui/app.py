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
USER_ID = f"st_user_{APP_NAME}"
ADK_SESSION_ID_KEY = f'adk_session_id_{APP_NAME}'
ADK_SERVICE_KEY = f'adk_service_{APP_NAME}'
ADK_RUNNER_KEY = f'adk_runner_{APP_NAME}'
MESSAGE_HISTORY_KEY = f"messages_{APP_NAME}"

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="GCP Agent Hub",
    page_icon="☁️"
    )

# --- ADK Imports ---
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("streamlit_app")
logger.info("Streamlit App Logger Initialized.")

# --- Apply nest_asyncio ---
try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully.")
except RuntimeError as e:
    if "cannot apply nest_asyncio" not in str(e):
         logger.error(f"Error applying nest_asyncio: {e}")

# --- ADK Initialization & Session Management ---
def get_runner_and_session_id():
    """
    Initializes or retrieves the ADK Runner and Session ID using Streamlit's session state.
    Handles creation and validation of the ADK session.
    """
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
            if ADK_SESSION_ID_KEY in st.session_state: del st.session_state[ADK_SESSION_ID_KEY]
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
            if ADK_SESSION_ID_KEY in st.session_state: del st.session_state[ADK_SESSION_ID_KEY]
            raise RuntimeError(f"Failed to validate/recreate ADK session {session_id}: {e}") from e

    return runner, session_id

# --- Async Runner Function ---
async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Runs a turn of the ADK agent asynchronously.
    """
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
            final_response_text = f"Sorry, an error occurred during agent execution: {e}"
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- ADK Run Async: Turn execution completed in {duration:.2f} seconds.")

    return final_response_text

# --- Sync Wrapper ---
def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> str:
    """
    Synchronous wrapper to call the async ADK runner function using asyncio.run.
    Handles potential runtime errors during async execution.
    """
    try:
        return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
    except RuntimeError as e:
        logger.exception("RuntimeError during asyncio.run in run_adk_sync:")
        return f"Error running agent task: {e}. Check logs."
    except Exception as e:
        logger.exception("Unexpected exception during run_adk_sync:")
        return f"An unexpected error occurred: {e}. Check logs."

# --- Initialize ADK ---
try:
    adk_runner, current_adk_session_id = get_runner_and_session_id()
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize ADK session: {e}", icon="❌")
    logger.exception("Critical ADK Initialization/Session Validation failed.")
    st.stop()

# --- Sidebar UI ---
with st.sidebar:
    st.markdown(
        """
        <h2 style='color:#FFFFFF; font-weight: 600; font-size: 1.5em; margin-bottom: 0px;'>
            <span style='color:#4285F4;'>G</span><span style='color:#EA4335;'>o</span><span style='color:#FBBC05;'>o</span><span style='color:#4285F4;'>g</span><span style='color:#34A853;'>l</span><span style='color:#EA4335;'>e</span> Cloud
        </h2>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    st.header("⚙️ Session Info")
    if current_adk_session_id:
        st.success(f"Session Active ✅")
    else:
         st.error("Session Inactive ❌")

    if st.button("🔄 Clear Chat & Reset Session"):
        logger.info(f"Clearing chat history and resetting session for user {USER_ID}, ADK session {current_adk_session_id}")
        st.session_state.pop(MESSAGE_HISTORY_KEY, None)
        st.session_state.pop(ADK_SESSION_ID_KEY, None)
        st.session_state.pop(ADK_RUNNER_KEY, None)
        st.session_state.pop(ADK_SERVICE_KEY, None)
        logger.info("Cleared ADK keys from st.session_state.")
        st.toast("Session Cleared!")
        st.rerun()

    st.divider()

    st.header("🤖 Agent Details")
    runner_instance = st.session_state.get(ADK_RUNNER_KEY)
    st.markdown(f"**Agent:** `{runner_instance.agent.name if runner_instance and runner_instance.agent else 'N/A'}`")
    st.markdown(f"**App:** `{APP_NAME}`")

    with st.expander("Show Full Session ID"):
        st.code(st.session_state.get(ADK_SESSION_ID_KEY, 'N/A'))

# --- Main Chat Interface UI ---
st.title("☁️ GCP Agent Hub")
st.caption("Powered by Google ADK")

st.info(
    """
    **What I can help with:**
    * **GCP Resources:** Manage Compute Engine VMs (create, list, start, stop, delete, details) and BigQuery Datasets (create).
    * **BigQuery Data:** Execute SQL queries against your BigQuery tables.
    * **GitHub:** Search for repositories and retrieve file contents.\n
    Ask me things like "list my VMs", "run a query to count users", or "find langchain repos on github".
    """,
    icon="ℹ️"
)

if MESSAGE_HISTORY_KEY not in st.session_state:
    st.session_state[MESSAGE_HISTORY_KEY] = [{"role": "assistant", "content": "Hello Samuel! As a fellow Googler passionate about AI/ML and Cloud, how can I assist you today?"}]

for message in st.session_state[MESSAGE_HISTORY_KEY]:
    avatar_icon = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar_icon):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about GCP resources, data, or GitHub..."):
    if not current_adk_session_id:
         st.error("Agent session ID could not be established. Cannot process request.", icon="❌")
    elif not adk_runner:
         st.error("ADK Runner is not available. Cannot process request.", icon="❌")
    else:
        st.session_state[MESSAGE_HISTORY_KEY].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            with st.spinner("Agent is processing..."):
                try:
                    agent_response = run_adk_sync(adk_runner, current_adk_session_id, USER_ID, prompt)
                    message_placeholder.markdown(agent_response)
                    st.session_state[MESSAGE_HISTORY_KEY].append({"role": "assistant", "content": agent_response})
                except Exception as e:
                    logger.exception("Error running ADK turn from Streamlit input:")
                    error_msg = f"An error occurred: {e}"
                    message_placeholder.error(error_msg, icon="🚨")
                    st.session_state[MESSAGE_HISTORY_KEY].append({"role": "assistant", "content": error_msg})