# agents/deepresearch/planner/agent.py
import os
from google.adk.agents import LlmAgent

agent_model = 'gemini-2.0-flash'

research_planner_agent = LlmAgent(
    name="ResearchPlanner",
    model=agent_model,
    description="Breaks down complex research queries into structured subtopics and outlines a research plan.",
    instruction=(
        "You are a Research Planner AI. Your task is to take a complex research query and break it down into a structured, actionable research plan.\n"
        "1.  **Decompose:** Break the main query into logical, well-defined subtopics covering key aspects (history, current state, impacts, challenges, future).\n"
        "2.  **Structure:** Organize the subtopics logically.\n"
        "3.  **Keywords:** For each subtopic, suggest relevant keywords for web searches.\n"
        "4.  **Output Format:** Present the plan clearly using markdown.\n\n"
        "**Example Output Structure:**\n"
        "```markdown\n"
        "# Research Plan: {Topic}\n\n"
        "## 1. {Subtopic 1}\n"
        "   - Keywords: {keyword1}, {keyword2}\n\n"
        "## 2. {Subtopic 2}\n"
        "   - Keywords: {keyword3}, {keyword4}\n\n"
        "...\n"
        "```\n"
        "Create a detailed research plan for the user's query."
    ),
    tools=[],
)
