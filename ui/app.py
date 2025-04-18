# adk-demo/ui/app.py
import streamlit as st
import logging
import time
import os
import asyncio
import nest_asyncio

APP_NAME = "gcp_multi_agent_demo_streamlit"
USER_ID = f"st_user_{APP_NAME}"
ADK_SESSION_ID_KEY = f'adk_session_id_{APP_NAME}'
ADK_SERVICE_KEY = f'adk_service_{APP_NAME}'
ADK_RUNNER_KEY = f'adk_runner_{APP_NAME}'
MESSAGE_HISTORY_KEY = f"messages_{APP_NAME}"

st.set_page_config(
    layout="wide",
    page_title="GCP Agent Hub",
    page_icon="☁️"
    )

try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai.types import Content, Part
except ImportError as e:
    st.exception(f"Failed to import agent modules or ADK components: {e}")
    st.error("Ensure project structure and requirements are correct. App cannot start.")
    st.stop()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nest_asyncio.apply()
    logger.info("nest_asyncio applied successfully.")
except RuntimeError as e:
    if "cannot apply nest_asyncio" not in str(e):
         logger.error(f"Error applying nest_asyncio: {e}")

def get_runner_and_session_id():
    if ADK_SERVICE_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new InMemorySessionService in st.session_state.")
        st.session_state[ADK_SERVICE_KEY] = InMemorySessionService()

    if ADK_RUNNER_KEY not in st.session_state:
        logger.info("--- ADK Init: Creating new Runner in st.session_state.")
        if 'meta_agent' not in globals():
             raise NameError("Fatal Error: meta_agent not imported or defined before Runner initialization.")
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

async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> tuple[str, str]:
    logger.info(f"\n--- ADK Run Async: Starting execution for session {session_id} ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"
    final_response_author = "assistant"
    start_time = time.time()

    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                 final_response_text = event.content.parts[0].text
                 final_response_author = event.author or "assistant"

    except Exception as e:
        if "Session not found" in str(e):
             logger.error(f"--- ADK Run Async: Confirmed 'Session not found' error for {session_id} / {user_id}")
             final_response_text = "Error: Agent session expired or was lost. Please try clearing the session and starting again."
             final_response_author = "error"
        else:
            logger.exception("--- ADK Run Async: !! EXCEPTION during agent execution:")
            final_response_text = f"Sorry, an error occurred during agent execution: {e}"
            final_response_author = "error"
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"--- ADK Run Async: Turn execution completed in {duration:.2f} seconds. Final author: {final_response_author}")

    return final_response_text, final_response_author

def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> tuple[str, str]:
    try:
        text, author = asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
        return text, author
    except RuntimeError as e:
        logger.exception("RuntimeError during asyncio.run in run_adk_sync:")
        return f"Error running agent task: {e}. Check logs.", "error"
    except Exception as e:
        logger.exception("Unexpected exception during run_adk_sync:")
        return f"An unexpected error occurred: {e}. Check logs.", "error"

try:
    adk_runner, current_adk_session_id = get_runner_and_session_id()
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize ADK session: {e}", icon="❌")
    logger.exception("Critical ADK Initialization/Session Validation failed.")
    st.stop()

AGENT_ICONS = {
    "user": "🧑‍💻",
    "MetaAgent": "🧠",
    "ResourceAgent": "☁️",
    "DataScienceAgent": "📊",
    "githubagent": "🐙",
    "MistralChatAgent": "🌬️",
    "assistant": "🤖",
    "error": "🚨"
}

with st.sidebar:
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        try:
            st.image("assets/google-cloud-logo.png", width=200)
        except FileNotFoundError:
            st.warning("Logo image not found.")
            st.header("☁️ Google Cloud")
    st.markdown(
    """
    <h2 style='text-align: center; color:#FFFFFF; font-weight: 600; font-size: 1.5em; margin-bottom: 0px;'>
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
    st.markdown(f"**Root Agent:** `{runner_instance.agent.name if runner_instance and runner_instance.agent else 'N/A'}`")
    st.markdown(f"**App:** `{APP_NAME}`")

    with st.expander("Show Full Session ID"):
        st.code(st.session_state.get(ADK_SESSION_ID_KEY, 'N/A'))


st.title("☁️ GCP Agent Hub")
st.caption("Powered by Google ADK")

st.info(
    """
    **What I can help with:**
    * **GCP Resources:** Manage Compute Engine VMs (create, list, start, stop, delete, details) and BigQuery Datasets (create).
    * **BigQuery Data:** Execute SQL queries against your BigQuery tables.
    * **GitHub:** Search for repositories and retrieve file contents.\n
    * **Chat:** General conversation with Mistral.\n
    Ask me things like "list my VMs", "run a query to count users", "find langchain repos on github", or "tell me a joke".
    """,
    icon="ℹ️"
)

# Initialize message history if it doesn't exist
if MESSAGE_HISTORY_KEY not in st.session_state:
    st.session_state[MESSAGE_HISTORY_KEY] = [{"author": "assistant", "content": "Hello dear Cloud enthusiast, how can I assist you today?"}]

for message in st.session_state[MESSAGE_HISTORY_KEY]:
    author = message.get("author", "assistant")
    icon = AGENT_ICONS.get(author, AGENT_ICONS["assistant"])
    with st.chat_message(name=author, avatar=icon):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask about GCP resources, data, GitHub, or just chat..."):
    if not current_adk_session_id or not adk_runner:
         st.error("Agent session could not be established. Cannot process request.", icon="❌")
    else:
        st.session_state[MESSAGE_HISTORY_KEY].append({"author": "user", "content": prompt})
        with st.spinner("Agent is processing..."): 
            try:
                agent_response_text, agent_response_author = run_adk_sync(
                    adk_runner, current_adk_session_id, USER_ID, prompt
                )
                st.session_state[MESSAGE_HISTORY_KEY].append({"author": agent_response_author, "content": agent_response_text})
            except Exception as e:
                logger.exception("Error running ADK turn from Streamlit input:")
                error_msg = f"An error occurred: {e}"
                st.session_state[MESSAGE_HISTORY_KEY].append({"author": "error", "content": error_msg})

        st.rerun()