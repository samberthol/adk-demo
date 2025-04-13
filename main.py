# main.py
# Import the main agent that the UI will interact with
from agents.meta.agent import meta_agent

print("Agent modules loaded.")

# Define the root agent for potential ADK CLI usage (adk run.)
root_agent = meta_agent