# main.py
from agents.meta.agent import meta_agent
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent

print("Agent modules loaded.")

root_agent = meta_agent