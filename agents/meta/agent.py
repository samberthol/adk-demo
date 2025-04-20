# agents/meta/agent.py
import os
import logging
from typing import Optional, List # Added List

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
# --- Imports needed for Callback --- START ---
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.genai.types import Content
# --- Imports needed for Callback --- END ---


# Import other sub-agents
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent
# Import from the nested 'llm_auditor' directory
from agents.llm_auditor.llm_auditor.agent import llm_auditor

logger = logging.getLogger(__name__)

# Get the model for the MetaAgent itself
agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')

# Define agent names
MISTRAL_AGENT_NAME = "MistralChatAgent"
LLM_AUDITOR_NAME = getattr(llm_auditor, 'name', 'llm_auditor')

# --- Callback Function to Filter Mistral History --- START ---
def _filter_mistral_history(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """
    Filters the history in the LLM request for MistralChatAgent
    to remove unsupported roles like 'tool'.
    """
    if not llm_request or not hasattr(llm_request, 'contents') or not llm_request.contents:
        logger.warning(f"[{callback_context.agent_name}] before_model_callback received empty request or no contents.")
        return

    logger.info(f"[{callback_context.agent_name}] Running before_model_callback to filter roles.")
    filtered_contents: List[Content] = []
    allowed_roles = {'user', 'assistant', 'system'}

    for content_item in llm_request.contents:
        if isinstance(content_item, Content) and hasattr(content_item, 'role'):
            if content_item.role in allowed_roles:
                filtered_contents.append(content_item)
            else:
                logger.debug(f"[{callback_context.agent_name}] Filtering out message with role: {content_item.role}")
        else:
            logger.warning(f"[{callback_context.agent_name}] Encountered unexpected item type in request contents: {type(content_item)}")
            filtered_contents.append(content_item)

    llm_request.contents = filtered_contents
    logger.info(f"[{callback_context.agent_name}] Finished filtering roles. Message count now: {len(llm_request.contents)}")

    return None
# --- Callback Function to Filter Mistral History --- END ---

# --- Instantiate Mistral Agent using LlmAgent and LiteLlm ---
mistral_agent = None
mistral_model_id = os.environ.get('MISTRAL_MODEL_ID')

if mistral_model_id:
    litellm_model_string = f"vertex_ai/{mistral_model_id}"
    logger.info(f"Configuring {MISTRAL_AGENT_NAME} using LiteLlm with model: {litellm_model_string}")
    try:
        mistral_agent = LlmAgent(
            name=MISTRAL_AGENT_NAME,
            model=LiteLlm(model=litellm_model_string),
            description="A conversational agent powered by Mistral via Vertex AI (using LiteLLM).",
            instruction="You are a helpful conversational AI assistant based on Mistral models. Respond directly to the user's query.",
            # Add the callback here
            before_model_callback=_filter_mistral_history # <-- ADDED THIS LINE
        )
        logger.info(f"Successfully configured {MISTRAL_AGENT_NAME} as LlmAgent with LiteLlm and history filter.") # Updated log
    except Exception as e:
        logger.error(f"Failed to configure {MISTRAL_AGENT_NAME} with LiteLlm: {e}", exc_info=True)
        mistral_agent = None
else:
    logger.warning(f"MISTRAL_MODEL_ID environment variable not set. {MISTRAL_AGENT_NAME} will not be available.")
    mistral_agent = None


# --- Build Active Sub-Agents List --- (Your existing logic is kept)
active_sub_agents = [resource_agent, data_science_agent, githubagent]

if mistral_agent:
    active_sub_agents.append(mistral_agent)
else:
    logger.warning(f"{MISTRAL_AGENT_NAME} could not be initialized and will not be available.")

if 'llm_auditor' in globals() and llm_auditor:
     active_sub_agents.append(llm_auditor)
     logger.info(f"Adding '{LLM_AUDITOR_NAME}' to sub-agents list.")
else:
     logger.error(f"LLM Auditor agent ('{LLM_AUDITOR_NAME}') object not found after import attempt. Check import path and Dockerfile copy step.")


# --- Define Meta Agent --- (Your existing logic is kept)
meta_agent = LlmAgent(
    name="MetaAgent",
    model=agent_model,
    description="A helpful assistant coordinating specialized agents (resources, data, GitHub, GCP Support/Auditor) and a general conversational agent (Mistral).",
    instruction=(
        "You are the primary assistant. Analyze the user's request.\n"
        "- If it involves managing cloud resources (like creating a VM or dataset), delegate the task to the 'ResourceAgent'.\n"
        "- If it involves querying data from BigQuery, delegate the task to the 'DataScienceAgent'.\n"
        "- If it involves searching GitHub or getting information from a GitHub repository, delegate the task to the 'githubagent'.\n"
        f"- If it asks for information about GCP services, documentation, code examples, or needs factual verification about GCP, delegate the task to the '{LLM_AUDITOR_NAME}' agent. Provide the user's question or the statement to verify as input to the auditor.\n"
        f"- If the request appears to be general conversation, requires summarization, explanation, brainstorming, or doesn't clearly fit other agents, delegate the task to the '{MISTRAL_AGENT_NAME}'.\n"
        "Clearly present the results from the specialist agents or the chat agent back to the user."
    ),
    sub_agents=active_sub_agents,
)