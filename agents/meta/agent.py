# agents/meta/agent.py
import os
import logging
from typing import Optional, List

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai.types import Content

# Import existing specialized agents
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent
from agents.deepresearch.coordinator.agent import deep_research_coordinator
from agents.langgraphagent.tools import langgraph_currency_tool

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
            # Keep items without a role (e.g., system instructions if not Content type)
            filtered_contents.append(content_item)
    llm_request.contents = filtered_contents


# Instantiate Mistral Agent
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
            before_model_callback=_filter_mistral_history # Apply history filtering
        )
    except Exception as e:
        logger.error(f"Failed to configure {MISTRAL_AGENT_NAME}: {e}", exc_info=True)
else:
    logger.warning(f"MISTRAL_MODEL_ID not set. {MISTRAL_AGENT_NAME} unavailable.")


# Assemble Sub-Agents List
active_sub_agents = [
    resource_agent,
    data_science_agent,
    githubagent,
    deep_research_coordinator,
]
if llm_auditor:
     active_sub_agents.append(llm_auditor)
if mistral_agent:
    active_sub_agents.append(mistral_agent)

meta_agent_tools = [
    langgraph_currency_tool,
]


# Define Meta Agent with updated instructions
meta_agent = LlmAgent(
    name="MetaAgent",
    model=agent_model,
    description="Coordinator for specialized agents and tools (resources, data, GitHub, currency, auditor, deep research) and a chat agent.",
    instruction=(
        "You are the primary assistant. Analyze the user's request carefully.\n"
        "**Routing Logic:**\n"
        "- If the request involves **in-depth research**, **investigation**, creating a **detailed report**, or analyzing a complex topic, **delegate the *entire* user request** to `DeepResearchCoordinatorAgent`. Examples: 'Investigate the impact of AI regulation', 'Create a detailed report on quantum computing advancements', 'Research the effects of climate change on agriculture'. Do not try to extract the topic yourself, just pass the full request.\n"
        "- If it involves managing cloud resources (VMs), delegate to `ResourceAgent`.\n"
        "- If it involves BigQuery data or datasets, delegate to `DataScienceAgent`.\n"
        "- If it involves GitHub repositories or files, delegate to `githubagent`.\n"
        f"- If it asks about GCP services, documentation, or needs GCP fact-checking, delegate to `{LLM_AUDITOR_NAME}`.\n"
        f"- If the request involves currency conversion or exchange rates, use the tool `langgraph_currency_a2a_tool_func`. Provide the user's query to the tool's `query` parameter.\n"
        f"- For general conversation, summarization, translation, or if no other specialist agent fits, delegate to `{MISTRAL_AGENT_NAME}`.\n\n"
        "**Execution:**\n"
        "- When delegating to another agent, simply invoke the agent transfer with the target agent's name.\n"
        "- When using a tool, call the specific tool function with the required parameters.\n"
        "- Present results clearly from tools or delegated agents back to the user."
    ),
    sub_agents=active_sub_agents,
    tools=meta_agent_tools
)