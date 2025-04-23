# agents/deepresearch/parallel/agent.py
import os
from google.adk.agents import ParallelAgent
from ..researcher.agent import researcher_agent

research_parallel_agent = ParallelAgent(
    name="ResearchParallelAgent",
    description="Executes multiple research tasks (subtopics) in parallel using instances of ResearcherAgent.",
    worker_agent=researcher_agent,
)
