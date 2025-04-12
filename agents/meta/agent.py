# agents/meta/agent.py
from google.adk.agents import LlmAgent 
from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent


meta_agent = LlmAgent(
    name="MetaAgent",
    model="gemini-2.0-flash",
    description="A helpful assistant that understands user requests and coordinates with specialized agents for resource management and data science tasks.",
    instruction="You are the primary assistant. Analyze the user's request. If it involves managing cloud resources (like creating a VM or dataset), delegate the task to the 'ResourceAgent'. If it involves querying data from BigQuery, delegate the task to the 'DataScienceAgent'. For general conversation, respond directly. Clearly present the results from the specialist agents back to the user.",
    sub_agents=[resource_agent, data_science_agent],
)