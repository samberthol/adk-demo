# agents/meta/agent.py
import os
import logging
from typing import Optional

# Import LlmAgent and LiteLlm wrapper
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm # <-- Import LiteLlm
# Remove imports related to old custom agent or callbacks
# from google.adk.agents.callback_context import CallbackContext
# from google.genai.types import Content, Part, FunctionCall
# from google.adk.models import LlmResponse
# from agents.mistral.agent import MistralVertexAgent # <-- Remove old import

# Import other sub-agents
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent

logger = logging.getLogger(__name__)

# Get the model for the MetaAgent itself
agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')

# Define the target agent name
MISTRAL_AGENT_NAME = "MistralChatAgent"

# --- Instantiate Mistral Agent using LlmAgent and LiteLlm ---
mistral_agent = None
mistral_model_id = os.environ.get('MISTRAL_MODEL_ID') # e.g., mistral-small-2503

if mistral_model_id:
    # Construct the LiteLLM model string for Vertex AI
    # Format: vertex_ai/<your-mistral-model-id>
    litellm_model_string = f"vertex_ai/{mistral_model_id}"
    logger.info(f"Configuring {MISTRAL_AGENT_NAME} using LiteLlm with model: {litellm_model_string}")
    try:
        mistral_agent = LlmAgent(
            name=MISTRAL_AGENT_NAME,
            # Pass LiteLlm instance to model parameter
            model=LiteLlm(model=litellm_model_string),
            description="A conversational agent powered by Mistral via Vertex AI (using LiteLLM).",
            instruction="You are a helpful conversational AI assistant based on Mistral models. Respond directly to the user's query."
            # LlmAgent handles interaction, no need for custom _run_async_impl
            # No callbacks needed for this approach
        )
        logger.info(f"Successfully configured {MISTRAL_AGENT_NAME} as LlmAgent with LiteLlm.")
    except Exception as e:
        # Catch potential errors during LlmAgent/LiteLlm init
        logger.error(f"Failed to configure {MISTRAL_AGENT_NAME} with LiteLlm: {e}", exc_info=True)
else:
    logger.warning(f"MISTRAL_MODEL_ID environment variable not set. {MISTRAL_AGENT_NAME} will not be available.")


# --- Build Active Sub-Agents List ---
active_sub_agents = [resource_agent, data_science_agent, githubagent]
if mistral_agent:
    active_sub_agents.append(mistral_agent)
else:
    logger.warning(f"{MISTRAL_AGENT_NAME} could not be initialized and will not be available.")


# --- Define Meta Agent ---
meta_agent = LlmAgent(
    name="MetaAgent",
    model=agent_model,
    description="A helpful assistant coordinating specialized agents (resources, data, GitHub) and a general conversational agent (Mistral).",
    instruction=(
        "You are the primary assistant. Analyze the user's request.\n"
        "- If it involves managing cloud resources (like creating a VM or dataset), delegate the task to the 'ResourceAgent'.\n"
        "- If it involves querying data from BigQuery, delegate the task to the 'DataScienceAgent'.\n"
        "- If it involves searching GitHub or getting information from a GitHub repository, delegate the task to the 'githubagent'.\n"
        f"- If the request appears to be general conversation, requires summarization, explanation, brainstorming, or doesn't clearly fit other agents, delegate the task to the '{MISTRAL_AGENT_NAME}'.\n"
        "Clearly present the results from the specialist agents or the chat agent back to the user."
    ),
    sub_agents=active_sub_agents,
    # No callback needed now
)