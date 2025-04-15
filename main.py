# main.py
# Import the main agent that the UI will interact with
from agents.meta.agent import meta_agent
# Optional: also import other agents if you might use them directly via CLI
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent # <-- Add import

print("Agent modules loaded.")

root_agent = meta_agent
