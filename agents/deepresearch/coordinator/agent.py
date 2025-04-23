# agents/deepresearch/coordinator/agent.py
import os
from google.adk.agents import LlmAgent
from ..planner.agent import research_planner_agent
from ..parallel.agent import research_parallel_agent
from ..loop.agent import analysis_loop_agent
from ..writer.agent import writing_agent
from ..editor.agent import editor_agent

agent_model = 'gemini-2.0-flash'

deep_research_coordinator = LlmAgent(
    name="DeepResearchCoordinatorAgent",
    model=agent_model,
    description="Orchestrates deep research using planning, parallel research, iterative analysis, writing, and editing.",
    instruction=(
        "You are the master coordinator for a multi-agent deep research team using advanced workflows.\n"
        "Available Sub-Agents:\n"
        "- `ResearchPlanner`: Creates a research plan.\n"
        "- `ResearchParallelAgent`: Takes a list of subtopic/task descriptions and runs `ResearcherAgent` for each in parallel, returning a list of results.\n"
        "- `AnalysisLoopAgent`: Takes aggregated research findings, runs `AnalysisAgent`, evaluates with `SatisfactionEvaluator`, and loops until analysis is 'sufficient' (max 3 loops).\n"
        "- `WritingAgent`: Writes an article from the final analysis.\n"
        "- `EditorAgent`: Edits the article based on the final analysis.\n\n"
        "**Execution Workflow:**\n"
        "Follow these steps for the user's query (research topic):\n"
        "1.  **Plan:** Delegate the query to `ResearchPlanner` -> Get the research plan (markdown).\n"
        "2.  **Prepare Parallel Tasks:** Parse the research plan. Create a list where each item is a string describing a single subtopic and its keywords (e.g., 'Subtopic: Historical Context and Evolution\\nKeywords: history AI regulation, AI ethics development'). This list will be the input for `ResearchParallelAgent`.\n"
        "3.  **Parallel Research:** Delegate the list of task descriptions to `ResearchParallelAgent` -> Get the list of research results (markdown snippets from each parallel researcher).\n"
        "4.  **Aggregate Research:** Combine the list of markdown results from `ResearchParallelAgent` into a single large markdown string. This aggregated text is the input for the analysis loop.\n"
        "5.  **Iterative Analysis:** Delegate the aggregated research string to `AnalysisLoopAgent` -> Get the final 'Critical Analysis Report' (markdown) once the loop determines it's sufficient.\n"
        "6.  **Write:** Delegate the final 'Critical Analysis Report' from `AnalysisLoopAgent` to `WritingAgent` -> Get the 'Draft Article' (markdown).\n"
        "7.  **Edit:** Delegate *both* the 'Draft Article' AND the final 'Critical Analysis Report' to `EditorAgent` -> Get the final 'Edited Article' (markdown with editor comments).\n"
        "8.  **Final Output:** Present the final 'Edited Article' from `EditorAgent`. Extract the list of source URLs from the final 'Critical Analysis Report' and append it under '## Sources Consulted'.\n\n"
        "**Output Structure:**\n"
        "```markdown\n"
        "{Paste the entire content of the Edited Article from EditorAgent here}\n\n"
        "## Sources Consulted\n"
        "- {URL 1 from Analysis Report}\n"
        "- {URL 2 from Analysis Report}\n"
        "- ...\n"
        "```\n"
        "**Important:** Manage the data flow carefully. Ensure correct parsing, list creation, aggregation, and delegation at each step."
    ),
    sub_agents=[
        research_planner_agent,
        research_parallel_agent,
        analysis_loop_agent,
        writing_agent,
        editor_agent,
    ],
    tools=[],
)
