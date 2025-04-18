# agents/meta/agent.py
# DEBUGGING VERSION - try/except removed

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

# Instantiate MistralVertexAgent (it will read its own env vars)
mistral_agent = None

# --- Debugging: Removed try/except block to see initialization errors directly ---
# Instantiate without passing config parameters
# If this fails, the application will now crash with a direct traceback
mistral_agent = MistralVertexAgent(
    name="MistralChatAgent", # Still useful to give it a distinct name
    description="A conversational agent powered by Mistral via Vertex AI.",
    instruction="You are a helpful conversational AI assistant based on Mistral models."
)
logger.info("Successfully instantiated MistralVertexAgent (MistralChatAgent)")
# --- End of Debugging Change ---


# Build the list of active sub-agents
active_sub_agents = [resource_agent, data_science_agent, githubagent]
# This check might be less relevant now as failure will likely crash above,
# but keep it for consistency if we add try/except back later.
if mistral_agent:
    active_sub_agents.append(mistral_agent)
else:
    # Log if it somehow didn't crash but agent is still None (shouldn't happen without try/except)
    logger.error("Mistral agent is None after instantiation attempt without try/except block. This is unexpected.")


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