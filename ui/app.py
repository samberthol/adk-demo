# adk-demo/ui/app.py
import streamlit as st
import logging
import time
import os
import asyncio
import nest_asyncio
from streamlit_mermaid import st_mermaid
from typing import Tuple, Set, List

# ADK Core Imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
# Agent class needed for type hints if used, but specific agents often imported directly
from google.adk.agents import Agent

# Agent Imports
# Import the main meta_agent instance (now configured with its direct tools)
from agents.meta.agent import meta_agent
# Import optional agents (used only for logging/display purposes here now)
from agents.meta.agent import llm_auditor, mistral_agent


APP_NAME = "gcp_multi_agent_demo_streamlit"
USER_ID = f"st_user_{APP_NAME}"
ADK_SESSION_ID_KEY = f'adk_session_id_{APP_NAME}'
ADK_SERVICE_KEY = f'adk_service_{APP_NAME}'
ADK_RUNNER_KEY = f'adk_runner_{APP_NAME}'
MESSAGE_HISTORY_KEY = f"messages_{APP_NAME}"
LAST_TURN_AUTHOR_KEY = f"last_author_{APP_NAME}"
ACTIVATED_AGENTS_KEY = f"activated_agents_{APP_NAME}" # Note: This might not show tool usage directly

st.set_page_config(
    layout="wide",
    page_title="GCP Agent Hub",
    page_icon="☁️"
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nest_asyncio.apply()
except RuntimeError as e:
    if "cannot apply nest_asyncio" not in str(e):
         logger.error(f"Error applying nest_asyncio: {e}")

def get_runner_and_session_id():
    """Initializes ADK services and runner, storing them in session state."""

    if ADK_SERVICE_KEY not in st.session_state:
        logger.info("Creating new InMemorySessionService.")
        st.session_state[ADK_SERVICE_KEY] = InMemorySessionService()

    session_service = st.session_state[ADK_SERVICE_KEY]

    if ADK_RUNNER_KEY not in st.session_state:
        logger.info("Creating new Runner.")

        # --- No longer need to instantiate bridge agent or dynamically update sub_agents ---
        # --- MetaAgent is imported fully configured from agents.meta.agent ---

        st.session_state[ADK_RUNNER_KEY] = Runner(
            agent=meta_agent, # Use the imported meta_agent instance
            app_name=APP_NAME,
            session_service=session_service
        )
        logger.info("Runner created and stored in session state.")

    runner: Runner = st.session_state[ADK_RUNNER_KEY]

    # --- Manage ADK Session ID ---
    if ADK_SESSION_ID_KEY not in st.session_state:
        session_id = f"st_session_{APP_NAME}_{int(time.time())}_{os.urandom(4).hex()}"
        st.session_state[ADK_SESSION_ID_KEY] = session_id
        logger.info(f"Generated new ADK session ID: {session_id}")
        try:
            # Pass initial state if needed, e.g., empty dict {}
            session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={})
            logger.info(f"Created session {session_id} in service.")
        except Exception as e:
            logger.exception(f"ERROR initially creating session {session_id}:")
            if ADK_SESSION_ID_KEY in st.session_state: del st.session_state[ADK_SESSION_ID_KEY]
            raise RuntimeError(f"Could not create initial ADK session {session_id}: {e}") from e
    else:
        session_id = st.session_state[ADK_SESSION_ID_KEY]
        try:
            existing = session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
            if not existing:
                logger.warning(f"Session {session_id} not found. Recreating (state lost).")
                session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id, state={})
        except Exception as e:
            logger.exception(f"Error checking/recreating session {session_id}:")
            if ADK_SESSION_ID_KEY in st.session_state: del st.session_state[ADK_SESSION_ID_KEY]
            raise RuntimeError(f"Failed to validate/recreate ADK session {session_id}: {e}") from e

    return runner, session_id

async def run_adk_async(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> Tuple[str, str, Set[str]]:
    logger.info(f"Starting ADK run_async for session {session_id}")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"
    final_response_author = "assistant"
    activated_agents_set = set() # This tracks agents, direct tool calls won't appear here

    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            author = event.author
            # Track activated sub-agents (tool calls by MetaAgent won't be tracked this way)
            if author and author != 'user' and author != runner.agent.name:
                activated_agents_set.add(author)
            # Capture the last text part as the response
            if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                 final_response_text = event.content.parts[0].text
                 final_response_author = author or runner.agent.name # Default to MetaAgent if author is None

    except Exception as e:
        logger.exception("Exception during agent execution:")
        final_response_text = f"Sorry, an error occurred: {e}"
        final_response_author = "error"
        activated_agents_set.add("error") # Add MetaAgent too?

    # If MetaAgent did the work (e.g., called a tool), add it to activated set for display
    if not activated_agents_set and final_response_author == runner.agent.name:
        activated_agents_set.add(runner.agent.name)
    elif final_response_author == "error":
         activated_agents_set.add("error")


    logger.info(f"ADK run_async completed. Final Author: {final_response_author}, Activated: {activated_agents_set}")
    return final_response_text, final_response_author, activated_agents_set

def run_adk_sync(runner: Runner, session_id: str, user_id: str, user_message_text: str) -> Tuple[str, str, Set[str]]:
    try:
        return asyncio.run(run_adk_async(runner, session_id, user_id, user_message_text))
    except Exception as e:
        logger.exception("Exception during run_adk_sync:")
        return f"An error occurred: {e}", "error", {"error"}

# Updated Icons (Removed bridge agent)
AGENT_ICONS = {
    "user": "🧑‍💻",
    "MetaAgent": "🧠",
    "ResourceAgent": "☁️",
    "DataScienceAgent": "📊",
    "githubagent": "🐙",
    # "A2ALangGraphCurrencyAgent": "💱", # Removed
    "MistralChatAgent": "🌬️",
    "llm_auditor": "🔎",
    "assistant": "🤖",
    "error": "🚨"
}

def generate_mermaid_syntax(root_agent_name: str, activated_agents: Set[str], last_author: str = None) -> str:
    if not root_agent_name: return "graph TD;\n  Error[ADK Runner not initialized];\n"
    mermaid_lines = ["graph TD"]
    nodes_to_draw = activated_agents.copy() if activated_agents else set()
    # Add root agent if it exists and isn't the only one active (unless it's an error)
    if root_agent_name and (len(nodes_to_draw) > 1 or (len(nodes_to_draw) == 1 and "error" not in nodes_to_draw)):
         nodes_to_draw.add(root_agent_name)
    elif not nodes_to_draw and last_author == root_agent_name: # Handle case where MetaAgent replies directly (e.g. after tool use)
         nodes_to_draw.add(root_agent_name)
    elif "error" in nodes_to_draw:
         nodes_to_draw.add(root_agent_name) # Show root agent on error too

    if not nodes_to_draw:
         mermaid_lines.append(f'    Idle["{AGENT_ICONS.get("assistant", "?")} Waiting..."]:::default')
    else:
        for name in nodes_to_draw:
            icon = AGENT_ICONS.get(name, '❓')
            mermaid_lines.append(f'    {name}["{icon} {name}"]')
        # Draw arrows only if MetaAgent is present and other nodes exist
        if root_agent_name in nodes_to_draw and len(nodes_to_draw) > 1:
            for name in nodes_to_draw:
                if name != root_agent_name and name != "error":
                    mermaid_lines.append(f'    {root_agent_name} --> {name}')
        mermaid_lines.append('    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#333')
        mermaid_lines.append('    classDef active fill:#D5E8D4,stroke:#82B366,stroke-width:2px,color:#000')
        for name in nodes_to_draw:
            # Highlight the actual author, even if it's MetaAgent after a tool call
            node_class = "active" if last_author == name else "default"
            mermaid_lines.append(f'    {name}:::{node_class}')
    return "\n".join(mermaid_lines) + "\n"

try:
    adk_runner, current_adk_session_id = get_runner_and_session_id()
    root_agent_name = adk_runner.agent.name if adk_runner and adk_runner.agent else None
except Exception as e:
    st.error(f"**Fatal Error:** Could not initialize ADK session/runner: {e}", icon="❌")
    logger.exception("Critical ADK Initialization failed.")
    root_agent_name = None
    adk_runner = None
    current_adk_session_id = None

with st.sidebar:
    img_col1, img_col2, img_col3 = st.columns([2, 4, 2])
    with img_col2:
        try: st.image("assets/google-cloud-logo.png", width=300)
        except FileNotFoundError: st.header("☁️ Google Cloud")
    st.markdown("<h2 style='text-align: center;'>Agent Hub</h2>", unsafe_allow_html=True)
    st.divider()

    st.header("⚙️ Session Info")
    if current_adk_session_id: st.success(f"Session Active ✅")
    else: st.error("Session Inactive ❌")
    if st.button("🔄 Clear Chat & Reset Session"):
        keys_to_pop = [MESSAGE_HISTORY_KEY, ADK_SESSION_ID_KEY, ADK_RUNNER_KEY, ADK_SERVICE_KEY, LAST_TURN_AUTHOR_KEY, ACTIVATED_AGENTS_KEY]
        for key in keys_to_pop: st.session_state.pop(key, None)
        st.toast("Session Cleared!")
        st.rerun()
    st.divider()

    st.header("🤖 Agent Activity")
    graph_col1, graph_col2, graph_col3 = st.columns([1, 4, 1])
    with graph_col2:
        if root_agent_name:
            last_author = st.session_state.get(LAST_TURN_AUTHOR_KEY)
            activated_agents = st.session_state.get(ACTIVATED_AGENTS_KEY)
            try:
                mermaid_syntax = generate_mermaid_syntax(root_agent_name, activated_agents, last_author)
                st_mermaid(mermaid_syntax, height=350)
            except Exception as e:
                 logger.error(f"Error displaying Mermaid chart: {e}", exc_info=True)
                 st.error("Error displaying activity.")
        else:
            st.warning("Runner not initialized.")
    with st.expander("Show Full Session ID"):
        st.code(st.session_state.get(ADK_SESSION_ID_KEY, 'N/A'))

st.title("☁️ GCP Agent Hub")
st.caption("Powered by Google ADK")

st.info(
    """
    **What I can help with:**
    * **GCP Resources:** Manage Compute Engine VMs and BigQuery Datasets.
    * **BigQuery Data:** Execute SQL queries.
    * **GitHub:** Search repositories and get file contents.
    * **Currency:** Get exchange rates (e.g., "1 USD to EUR?").
    * **GCP Support:** Answer questions about GCP services and documentation.
    * **Chat:** General conversation.
    """,
    icon="ℹ️"
)

if MESSAGE_HISTORY_KEY not in st.session_state:
    st.session_state[MESSAGE_HISTORY_KEY] = [{"author": "assistant", "content": "Hello! How can I assist with GCP or other tasks today?"}]

for message in st.session_state[MESSAGE_HISTORY_KEY]:
    author = message.get("author", "assistant")
    icon = AGENT_ICONS.get(author, AGENT_ICONS["assistant"])
    with st.chat_message(name=author, avatar=icon):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about GCP, GitHub, currency, or just chat..."):
    if not current_adk_session_id or not adk_runner:
         st.error("ADK session/runner not available. Cannot process request.", icon="❌")
    else:
        st.session_state[MESSAGE_HISTORY_KEY].append({"author": "user", "content": prompt})
        with st.chat_message(name="user", avatar=AGENT_ICONS["user"]):
             st.markdown(prompt)

        with st.spinner("Agent is processing..."):
            agent_response_text, agent_response_author, activated_agents_set = run_adk_sync(
                adk_runner, current_adk_session_id, USER_ID, prompt
            )
            st.session_state[LAST_TURN_AUTHOR_KEY] = agent_response_author
            st.session_state[ACTIVATED_AGENTS_KEY] = activated_agents_set
            st.session_state[MESSAGE_HISTORY_KEY].append({"author": agent_response_author, "content": agent_response_text})

        st.rerun()
