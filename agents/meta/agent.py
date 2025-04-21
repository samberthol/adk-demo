# agents/meta/agent.py
import os
import logging
from typing import Optional, List

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai.types import Content
# Required for type hinting and eventual instantiation of the new agent
from google.adk.sessions import SessionService
from agents.langgraphagent.agent import A2ALangGraphCurrencyAgent


# Import other sub-agents
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent

# Assuming llm_auditor was correctly copied
try:
    from agents.llm_auditor.llm_auditor.agent import llm_auditor
    LLM_AUDITOR_LOADED = True
    LLM_AUDITOR_NAME = getattr(llm_auditor, 'name', 'llm_auditor')
except ImportError:
    logging.error("Failed to import llm_auditor agent.", exc_info=True)
    llm_auditor = None
    LLM_AUDITOR_LOADED = False
    LLM_AUDITOR_NAME = "llm_auditor (Not Loaded)"

logger = logging.getLogger(__name__)

agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')
MISTRAL_AGENT_NAME = "MistralChatAgent"


# Callback Function to Filter Mistral History (if used)
def _filter_mistral_history(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    if not llm_request or not hasattr(llm_request, 'contents') or not llm_request.contents:
        return

    filtered_contents: List[Content] = []
    allowed_roles = {'user', 'assistant', 'system'}

    for content_item in llm_request.contents:
        if isinstance(content_item, Content) and hasattr(content_item, 'role'):
            if content_item.role in allowed_roles:
                filtered_contents.append(content_item)
        else:
            filtered_contents.append(content_item)

    llm_request.contents = filtered_contents


# Instantiate Mistral Agent (if configured)
mistral_agent = None
mistral_model_id = os.environ.get('MISTRAL_MODEL_ID')
if mistral_model_id:
    litellm_model_string = f"vertex_ai/{mistral_model_id}"
    try:
        mistral_agent = LlmAgent(
            name=MISTRAL_AGENT_NAME,
            model=LiteLlm(model=litellm_model_string),
            description="Conversational agent powered by Mistral.",
            instruction="Respond directly to the user's query.",
            before_model_callback=_filter_mistral_history
        )
    except Exception as e:
        logger.error(f"Failed to configure {MISTRAL_AGENT_NAME}: {e}", exc_info=True)
else:
    logger.warning(f"MISTRAL_MODEL_ID not set. {MISTRAL_AGENT_NAME} unavailable.")


# --- IMPORTANT ---
# The A2ALangGraphCurrencyAgent needs the SessionService during instantiation.
# This instantiation should happen where the Runner is created (e.g., ui/app.py)
# We define a placeholder here to include it in the list structure.
# Replace this with the actual instance in your Runner setup.
a2a_langgraph_currency_agent: Optional[A2ALangGraphCurrencyAgent] = None
A2A_LANGGRAPH_AGENT_NAME = "A2ALangGraphCurrencyAgent" # Use the name defined in its class


# --- Assemble Sub-Agents List ---
# Note: The actual list passed to MetaAgent might be constructed dynamically
# where the Runner is initialized, ensuring a2a_langgraph_currency_agent is instantiated.
active_sub_agents = [
    resource_agent,
    data_science_agent,
    githubagent,
    # Add instantiated a2a_langgraph_currency_agent here in the final setup
]
if llm_auditor:
     active_sub_agents.append(llm_auditor)
if mistral_agent:
    active_sub_agents.append(mistral_agent)
# Example of adding the placeholder (remove if list constructed elsewhere):
# if a2a_langgraph_currency_agent: # This check is illustrative
#    active_sub_agents.append(a2a_langgraph_currency_agent)


# --- Define Meta Agent ---
# Note: The 'sub_agents' list might be passed dynamically during instantiation
meta_agent = LlmAgent(
    name="MetaAgent",
    model=agent_model,
    description="Coordinator for specialized agents (resources, data, GitHub, currency, auditor) and a chat agent.",
    instruction=(
        "You are the primary assistant. Analyze the user's request.\n"
        "- If it involves managing cloud resources (VMs), delegate to 'ResourceAgent'.\n"
        "- If it involves BigQuery data or datasets, delegate to 'DataScienceAgent'.\n"
        "- If it involves GitHub repositories or files, delegate to 'githubagent'.\n"
        f"- If it asks about GCP services, documentation, or needs GCP fact-checking, delegate to '{LLM_AUDITOR_NAME}'.\n"
        # --- ADDED ROUTING for LangGraph Agent ---
        f"- If the request involves currency conversion, exchange rates, or mentions the 'currency agent', delegate to '{A2A_LANGGRAPH_AGENT_NAME}'.\n"
        # --- End LangGraph Routing ---
        f"- For general conversation, summarization, or if no other agent fits, delegate to '{MISTRAL_AGENT_NAME}'.\n"
        "Present results clearly."
    ),
    # Pass the list of agent instances. This might be done where MetaAgent is instantiated.
    sub_agents=active_sub_agents,
)

