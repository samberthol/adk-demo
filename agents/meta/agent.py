# agents/meta/agent.py
import os
import logging
from google.adk.agents import LlmAgent
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent
# Import the Mistral Agent
from agents.mistral.agent import MistralVertexAgent

logger = logging.getLogger(__name__)

# Get the model for the MetaAgent itself
agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')

# Instantiate MistralVertexAgent (it will read its own env vars and init SDK)
mistral_agent = None
# --- Reinstate try/except block for robustness ---
try:
    # Instantiate without passing config parameters (Agent handles its own config)
    mistral_agent = MistralVertexAgent(
        name="MistralChatAgent",
        description="A conversational agent powered by Mistral via Vertex AI.",
        instruction="You are a helpful conversational AI assistant based on Mistral models."
    )
    # Log success only if instantiation passes __init__ checks
    logger.info("Successfully instantiated MistralVertexAgent (MistralChatAgent)")
except ValueError as e:
    # Catch errors if required env vars are missing within the agent's __init__
    logger.warning(f"Could not instantiate MistralVertexAgent - Missing Env Var(s): {e}")
except RuntimeError as e:
    # Catch errors related to SDK/Endpoint initialization inside __init__
    logger.error(f"Could not instantiate MistralVertexAgent - SDK/Endpoint Init Error: {e}", exc_info=False) # exc_info=False to avoid duplicate traceback
except Exception as e:
    # Catch any other unexpected initialization errors
    logger.error(f"Unexpected error instantiating MistralVertexAgent: {e}", exc_info=True)
# --- End of reinstated try/except ---


# Build the list of active sub-agents
active_sub_agents = [resource_agent, data_science_agent, githubagent]
if mistral_agent:
    active_sub_agents.append(mistral_agent)


# Define Meta Agent
meta_agent = LlmAgent(
    name="MetaAgent",
    model=agent_model,
    description="A helpful assistant coordinating specialized agents (resources, data, GitHub) and a general conversational agent (Mistral).",
    instruction=(
        "You are the primary assistant. Analyze the user's request.\n"
        "- If it involves managing cloud resources (like creating a VM or dataset), delegate the task to the 'ResourceAgent'.\n"
        "- If it involves querying data from BigQuery, delegate the task to the 'DataScienceAgent'.\n"
        "- If it involves searching GitHub or getting information from a GitHub repository, delegate the task to the 'githubagent'.\n"
        "- If the request appears to be general conversation to the Mistral Agent, requires summarization, explanation, brainstorming, or doesn't clearly fit other agents, delegate the task to the 'MistralChatAgent'.\n"
        "Clearly present the results from the specialist agents or the chat agent back to the user."
    ),
    sub_agents=active_sub_agents,
)