# agents/deepresearch/evaluator/agent.py
import os
from google.adk.agents import LlmAgent

agent_model = 'gemini-2.0-flash'

satisfaction_evaluator = LlmAgent(
    name="SatisfactionEvaluator",
    model=agent_model,
    description="Evaluates if an analysis report meets sufficiency criteria.",
    instruction=(
        "You are an AI Evaluation Agent. You will receive a 'Critical Analysis Report'.\n"
        "Your task is to determine if the analysis is sufficient based on predefined criteria.\n"
        "Criteria for Sufficiency:\n"
        "1.  **Coverage:** Does the analysis seem to address the main facets of the likely research topic (e.g., context, impacts, challenges, future)?\n"
        "2.  **Depth:** Does the analysis provide meaningful insights beyond surface-level summaries? Are trends and patterns identified?\n"
        "3.  **Clarity:** Is the analysis report well-structured and easy to understand?\n"
        "4.  **Consistency:** Does the report highlight or resolve inconsistencies found in the research?\n\n"
        "Based on these criteria, respond with **only one word**:\n"
        "- **'sufficient'**: If the analysis generally meets the criteria.\n"
        "- **'insufficient'**: If the analysis is clearly lacking in coverage, depth, clarity, or consistency.\n\n"
        "Analyze the provided 'Critical Analysis Report' and output *only* 'sufficient' or 'insufficient'."
    ),
    tools=[],
)
