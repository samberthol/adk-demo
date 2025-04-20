# agents/meta/agent.py
import os
import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Import other sub-agents
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent
from agents.llm_auditor.agent import llm_auditor

logger = logging.getLogger(__name__)

# Get the model for the MetaAgent itself
agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')

# Define agent names
MISTRAL_AGENT_NAME = "MistralChatAgent"
# Get the actual name defined in the llm_auditor module, default if needed
LLM_AUDITOR_NAME = getattr(llm_auditor, 'name', 'llm_auditor')

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
            instruction="You are a helpful conversational AI assistant based on Mistral models. Respond directly to the user's query."
        )
        logger.info(f"Successfully configured {MISTRAL_AGENT_NAME} as LlmAgent with LiteLlm.")
    except Exception as e:
        logger.error(f"Failed to configure {MISTRAL_AGENT_NAME} with LiteLlm: {e}", exc_info=True)
        # Set to None if init fails, handle below
        mistral_agent = None
else:
    logger.warning(f"MISTRAL_MODEL_ID environment variable not set. {MISTRAL_AGENT_NAME} will not be available.")
    mistral_agent = None


# --- Build Active Sub-Agents List ---
# Start with existing agents that should always be present
active_sub_agents = [resource_agent, data_science_agent, githubagent]

# Add Mistral agent if successfully configured
if mistral_agent:
    active_sub_agents.append(mistral_agent)
else:
    logger.warning(f"{MISTRAL_AGENT_NAME} could not be initialized and will not be available.")

# Add LLM Auditor agent (llm_auditor instance is imported directly)
if llm_auditor:
     active_sub_agents.append(llm_auditor)
     logger.info(f"Adding '{LLM_AUDITOR_NAME}' to sub-agents list.")
else:
     # This case should ideally not happen if the import works,
     # but added for safety incase llm_auditor is None for some reason
     logger.warning(f"LLM Auditor agent ('{LLM_AUDITOR_NAME}') not loaded correctly and will not be available.")


# --- Define Meta Agent (Update Instructions) ---
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