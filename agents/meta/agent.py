# agents/meta/agent.py
import os
from google.adk.agents import LlmAgent
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
# --- Import the new githubagent ---
from agents.githubagent.agent import githubagent

# Read model name from environment variable, with a fallback
agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')

meta_agent = LlmAgent(
    name="MetaAgent",
    model=agent_model,
    # --- Updated description ---
    description="A helpful assistant that understands user requests and coordinates with specialized agents for resource management, data science tasks, and GitHub interactions.",
    # --- Updated instruction ---
    instruction=(
        "You are the primary assistant. Analyze the user's request.\n"
        "- If it involves managing cloud resources (like creating a VM or dataset), delegate the task to the 'ResourceAgent'.\n"
        "- If it involves querying data from BigQuery, delegate the task to the 'DataScienceAgent'.\n"
        "- If it involves searching GitHub or getting information from a GitHub repository, delegate the task to the 'githubagent'.\n" # <-- New rule
        "- For general conversation, respond directly.\n"
        "Clearly present the results from the specialist agents back to the user."
    ),
    # --- Add githubagent to sub_agents ---
    sub_agents=[resource_agent, data_science_agent, githubagent],
)