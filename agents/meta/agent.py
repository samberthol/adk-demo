# agents/meta/agent.py
import os
from google.adk.agents import BaseAgent # Import BaseAgent if needed by custom agent
# Remove LlmAgent import if no longer used directly here
# from google.adk.agents import LlmAgent

# Import your custom agent
from agents.custom.vertex_mistral_agent import VertexMistralAgent

# Keep imports for sub-agents (assuming they don't change)
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent

# --- Get environment variables ---
# Use the full path for AGENT_MODEL_NAME in your .env / trigger config
agent_model_name = os.environ.get('AGENT_MODEL_NAME')
project_id = os.environ.get('GCP_PROJECT_ID')
location = os.environ.get('REGION')

if not all([agent_model_name, project_id, location]):
    raise ValueError("Missing required environment variables: AGENT_MODEL_NAME, GCP_PROJECT_ID, REGION")

# --- Instantiate the custom agent ---
meta_agent = VertexMistralAgent(
    name="MetaAgent",
    model_name=agent_model_name, # Pass the full path
    project=project_id,
    location=location,
    description="A helpful assistant that understands user requests and coordinates with specialized agents for resource management, data science tasks, and GitHub interactions.",
    instruction=(
        "You are the primary assistant using a Mistral model on Vertex AI. Analyze the user's request.\n"
        "- If it involves managing cloud resources (like creating a VM or dataset), delegate the task to the 'ResourceAgent'.\n"
        "- If it involves querying data from BigQuery, delegate the task to the 'DataScienceAgent'.\n"
        "- If it involves searching GitHub or getting information from a GitHub repository, delegate the task to the 'githubagent'.\n"
        "- For general conversation, respond directly.\n"
        "Clearly present the results from the specialist agents back to the user."
        # NOTE: If this agent needs to MAKE DECISIONS about tool use/delegation,
        # the VertexMistralAgent needs to support parsing tool calls from the LLM response.
        # The current implementation only extracts text. Tool handling adds complexity.
    ),
    # Pass sub-agents if your custom agent knows how to handle delegation
    # The BaseAgent doesn't automatically handle sub_agents like LlmAgent does.
    # You might need to implement delegation logic within VertexMistralAgent's run_async
    # OR keep MetaAgent as LlmAgent but configure it to use a *different* LLM
    # just for routing, while other agents use Mistral? This gets complex.

    # For now, assuming MetaAgent primarily does generation/routing based on LLM response text:
    # sub_agents=[resource_agent, data_science_agent, githubagent], # This might not work directly with BaseAgent inheritance
)

# --- IMPORTANT ---
# If MetaAgent NEEDS tool use / delegation capabilities like LlmAgent,
# VertexMistralAgent needs significant enhancements:
# 1. Include tool definitions in the prompt sent to the LLM.
# 2. Parse the LLM response for tool call requests (not just text).
# 3. Yield ToolRequestEvent instead of FinalResponseEvent when a tool is requested.
# 4. Handle ToolResponseEvent coming back after a tool runs.
# This essentially reimplements parts of LlmAgent's logic.
# Consider the exact role of MetaAgent - if it only needs text generation, the above is simpler.