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
        "Your task is to determine if the analysis is sufficient based on predefined criteria.\n\n"
        "Criteria for Sufficiency:\n"
        "1.  **Coverage:** Does the analysis seem to comprehensively address the main facets and subtopics of the likely research topic (e.g., context, technical details, comparisons, applications, impacts, challenges, future)?\n"
        "2.  **Depth:** Does the analysis provide meaningful insights, supported by details from the research, beyond surface-level summaries? Are trends, patterns, and significant differentiators identified and elaborated upon?\n"
        "3.  **Clarity & Structure:** Is the analysis report well-structured, using clear headings and language, making it easy to understand the main points and conclusions?\n"
        "4.  **Consistency & Critical Assessment:** Does the report highlight or resolve inconsistencies found in the research? Does it critically assess the information where appropriate (e.g., mentioning source diversity, potential gaps)?\n"
        "5.  **Source Referencing:** Does the report include a 'References' section that appears to list the sources mentioned in the underlying research findings?\n"
        "6.  **Comparative Detail (if applicable to the topic):** If the research topic involves comparison (e.g., between two technologies), does the analysis go beyond surface-level statements and delve into specific points of comparison, detailing similarities, differences, pros, and cons, backed by evidence from the findings?\n\n"
        "Based on these criteria, respond with **only one word**:\n"
        "- **'sufficient'**: If the analysis generally meets ALL the applicable criteria well.\n"
        "- **'insufficient'**: If the analysis is clearly lacking in one or more significant criteria (e.g., poor coverage, superficial depth, missing comparative detail where expected, unclear structure).\n\n"
        "Analyze the provided 'Critical Analysis Report' and output *only* 'sufficient' or 'insufficient'."
    ),
    tools=[],
)