# agents/deepresearch/coordinator/agent.py
import os
import logging
from typing import Dict, Any, Optional, List, Tuple

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools import ToolContext
# Corrected import: Import Content, Part, and ToolConfig from google.genai.types
# Also import the necessary sub-classes/enums for ToolConfig
from google.genai.types import Content, Part, ToolConfig, FunctionLibrary

from ..planner.agent import research_planner_agent
from ..researcher.agent import researcher_agent
from ..loop.agent import analysis_loop_agent
from ..writer.agent import writing_agent
from ..editor.agent import editor_agent

# --- State Keys ---
COORD_STEP_KEY = "coord_step"
RESEARCH_PLAN_KEY = "research_plan"
RESEARCH_FINDINGS_KEY = "research_findings"
AGGREGATED_FINDINGS_KEY = "aggregated_findings"
ANALYSIS_DOSSIER_KEY = "analysis_dossier"
DRAFT_ARTICLE_KEY = "draft_article"
EDITED_ARTICLE_KEY = "edited_article"
CURRENT_RESEARCH_TASK_INDEX_KEY = "current_research_task_index"
PARSED_RESEARCH_TASKS_KEY = "parsed_research_tasks"
INITIAL_QUERY_KEY = "initial_query"

# Define the sequence of steps
WORKFLOW_STEPS: Dict[int, Dict[str, Any]] = {
    1: {"action": "call_planner", "agent": research_planner_agent.name},
    2: {"action": "execute_research_tasks", "agent": researcher_agent.name},
    3: {"action": "aggregate_findings"},
    4: {"action": "call_analyzer", "agent": analysis_loop_agent.name},
    5: {"action": "call_writer", "agent": writing_agent.name},
    6: {"action": "call_editor", "agent": editor_agent.name},
    7: {"action": "assemble_output"},
}
# --- End State Keys ---

agent_model = 'gemini-2.5-pro-preview-03-25'
logger = logging.getLogger(__name__)

# --- Callback Functions ---

def before_coord_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Injects current step information into the LLM request."""
    state = callback_context.session_context.state
    current_step = state.get(COORD_STEP_KEY, 1)

    if current_step == 1 and INITIAL_QUERY_KEY not in state:
        user_query = "User query not found in history."
        if llm_request.contents:
            for content in reversed(llm_request.contents):
                 if hasattr(content, 'role') and content.role == 'user' and hasattr(content, 'parts') and content.parts:
                     text_part = next((p.text for p in content.parts if hasattr(p, 'text')), None)
                     if text_part:
                         user_query = text_part
                         break
        state[INITIAL_QUERY_KEY] = user_query
        logger.info(f"Coordinator state: Stored initial query: '{user_query[:100]}...'")

    step_info = WORKFLOW_STEPS.get(current_step)
    step_description = f"Executing Step {current_step}: {step_info['action']}" if step_info else f"Unknown Step {current_step}"
    state_prompt = f"SYSTEM_INFO: You are currently on step {current_step} of the research workflow. Your goal is to execute this step based on the overall instructions and stored state. Step details: {step_description}."
    system_message = Content(role='system', parts=[Part(text=state_prompt)])

    inserted = False
    if llm_request.contents:
         for i in range(len(llm_request.contents) - 1, -1, -1):
             content = llm_request.contents[i]
             if hasattr(content, 'role') and content.role in ('user', 'assistant'):
                 llm_request.contents.insert(i + 1, system_message)
                 inserted = True
                 break
    if not inserted:
         llm_request.contents.insert(0, system_message)

    logger.info(f"Coordinator state: BEFORE model call. Current Step: {current_step}. Prompt modified.")


def after_coord_model(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    """Processes the LLM response, updates state, and potentially advances the step."""
    state = callback_context.session_context.state
    current_step = state.get(COORD_STEP_KEY, 1)
    step_info = WORKFLOW_STEPS.get(current_step)
    initial_query = state.get(INITIAL_QUERY_KEY, 'Missing Query')

    logger.info(f"Coordinator state: AFTER model call. Current Step: {current_step}. Got LLM Response.")

    if not step_info:
        logger.error(f"Coordinator state: Invalid step number {current_step} found in state.")
        return

    next_step = current_step
    step_completed = False

    # --- Process based on current step ---

    if llm_response.function_calls:
        func_call = llm_response.function_calls[0]
        logger.info(f"Coordinator state: LLM requested function call: {func_call.name}")
        expected_agent_name = step_info.get("agent")
        is_transfer = func_call.name == 'transfer_to_agent'
        func_args = func_call.args if hasattr(func_call, 'args') else {}

        if is_transfer and expected_agent_name and func_args.get('agent_name') == expected_agent_name:
            logger.info(f"Coordinator state: Correct agent transfer requested for step {current_step}: {expected_agent_name}")
            if current_step == 2:
                task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                logger.info(f"Coordinator state: Transferring to Researcher for task index {task_index}.")
            step_completed = False # Waiting for sub-agent result
        elif func_call.name != 'transfer_to_agent':
             logger.warning(f"Coordinator state: LLM called unexpected tool '{func_call.name}' during step {current_step}.")
        else:
             logger.warning(f"Coordinator state: LLM requested transfer to wrong agent '{func_args.get('agent_name')}' during step {current_step}. Expected '{expected_agent_name}'.")

    elif llm_response.function_responses:
        func_response = llm_response.function_responses[0]
        func_response_name = getattr(func_response, 'name', 'UnknownFunction')
        logger.info(f"Coordinator state: Received function response for: {func_response_name}")
        expected_agent_name = step_info.get("agent")

        if func_response_name == expected_agent_name:
            logger.info(f"Coordinator state: Received result from expected agent '{expected_agent_name}' for step {current_step}.")
            response_data = func_response.response if isinstance(func_response.response, dict) else {}
            response_content = response_data.get("content", "Error: No content in response")

            if isinstance(response_content, str) and ("error" in response_content.lower() or "failed" in response_content.lower()):
                 logger.error(f"Coordinator state: Sub-agent {expected_agent_name} reported an error: {response_content}")
                 step_completed = False
            else:
                # Store result based on step
                if current_step == 1: # Planner result
                    state[RESEARCH_PLAN_KEY] = response_content
                    logger.info("Coordinator state: Stored research plan.")
                    try: # Parse Plan
                        tasks = []
                        current_task_lines = []
                        lines = response_content.strip().splitlines()
                        plan_title = lines[0] if lines and lines[0].startswith("#") else "Research Plan"
                        start_index = 1 if lines and lines[0].startswith("#") else 0
                        for line in lines[start_index:]:
                            stripped_line = line.strip()
                            is_separator = stripped_line == '---'
                            is_new_task = stripped_line.startswith("**Task_ID:**")
                            if (is_separator or is_new_task) and current_task_lines:
                                task_text = "\n".join(current_task_lines).strip()
                                if "**Task_ID:**" in task_text: tasks.append(task_text)
                                current_task_lines = []
                            if not is_separator: current_task_lines.append(line)
                        if current_task_lines:
                             task_text = "\n".join(current_task_lines).strip()
                             if "**Task_ID:**" in task_text: tasks.append(task_text)
                        if not tasks:
                             logger.error("Coordinator state: Failed to parse any valid tasks from the plan.")
                             step_completed = False
                        else:
                             state[PARSED_RESEARCH_TASKS_KEY] = tasks
                             state[CURRENT_RESEARCH_TASK_INDEX_KEY] = 0
                             logger.info(f"Coordinator state: Parsed {len(tasks)} research tasks from plan titled '{plan_title}'.")
                             step_completed = True
                    except Exception as e:
                         logger.error(f"Coordinator state: Error parsing research plan: {e}", exc_info=True)
                         step_completed = False
                elif current_step == 2: # Researcher result
                    task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                    if RESEARCH_FINDINGS_KEY not in state: state[RESEARCH_FINDINGS_KEY] = []
                    if isinstance(state.get(RESEARCH_FINDINGS_KEY), list):
                        state[RESEARCH_FINDINGS_KEY].append(response_content)
                        logger.info(f"Coordinator state: Stored finding for task index {task_index}.")
                    else:
                        logger.error(f"Coordinator state: {RESEARCH_FINDINGS_KEY} is not a list. Cannot store finding.")
                        state[RESEARCH_FINDINGS_KEY] = [response_content]
                    next_task_index = task_index + 1
                    parsed_tasks = state.get(PARSED_RESEARCH_TASKS_KEY, [])
                    if next_task_index < len(parsed_tasks):
                        state[CURRENT_RESEARCH_TASK_INDEX_KEY] = next_task_index
                        logger.info(f"Coordinator state: Advancing to next research task index {next_task_index}.")
                        step_completed = False # Stay in step 2
                    else:
                        logger.info("Coordinator state: All research tasks completed.")
                        step_completed = True # Ready for step 3
                elif current_step == 4: # Analyzer result
                    state[ANALYSIS_DOSSIER_KEY] = response_content
                    logger.info("Coordinator state: Stored analysis dossier.")
                    step_completed = True
                elif current_step == 5: # Writer result
                    state[DRAFT_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state: Stored draft article.")
                    step_completed = True
                elif current_step == 6: # Editor result
                    state[EDITED_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state: Stored edited article.")
                    step_completed = True
                else:
                     logger.warning(f"Coordinator state: Received function response for unexpected step {current_step}")
                     step_completed = False
                if step_completed: next_step = current_step + 1
        else:
             logger.warning(f"Coordinator state: Received function response for unexpected function '{func_response_name}'. Expected '{expected_agent_name}'.")

    elif llm_response.text:
        logger.info("Coordinator state: LLM provided direct text response.")
        if current_step == 3: # Aggregate Findings
            logger.info("Coordinator state: Executing Step 3: Aggregate Findings.")
            findings = state.get(RESEARCH_FINDINGS_KEY, [])
            if isinstance(findings, list) and findings:
                aggregated = "\n\n---\n\n".join(str(f) for f in findings)
                state[AGGREGATED_FINDINGS_KEY] = aggregated
                logger.info(f"Coordinator state: Aggregated {len(findings)} findings.")
                step_completed = True
                next_step = current_step + 1
            else:
                logger.error(f"Coordinator state: Cannot aggregate findings, none found or not a list in state.")
                step_completed = False
        elif current_step == 7: # Assemble Final Output
            logger.info("Coordinator state: Executing Step 7: Assemble Final Output.")
            edited_article = state.get(EDITED_ARTICLE_KEY)
            analysis_dossier = state.get(ANALYSIS_DOSSIER_KEY)
            if not edited_article:
                 logger.error("Coordinator state: Cannot assemble final output, edited article missing.")
                 llm_response.text = "Error: Failed to generate the final article."
                 step_completed = False
                 next_step = 99 # End with error
            else:
                 references = "## Sources Consulted\nNo sources were listed in the analysis dossier."
                 if analysis_dossier and isinstance(analysis_dossier, str):
                      try:
                           ref_header = "## VI. Master Reference List"
                           ref_start_index = analysis_dossier.find(ref_header)
                           if ref_start_index != -1:
                                content_start = analysis_dossier.find('\n', ref_start_index) + 1
                                if content_start > 0:
                                     ref_content_block = analysis_dossier[content_start:]
                                     next_h2_index = ref_content_block.find("\n## ")
                                     if next_h2_index != -1: ref_content_block = ref_content_block[:next_h2_index]
                                     references = f"## Sources Consulted\n{ref_content_block.strip()}"
                                else: logger.warning("Coordinator state: Found reference header but no content after it.")
                           else: logger.warning("Coordinator state: 'Master Reference List' section header not found in dossier.")
                      except Exception as e: logger.error(f"Coordinator state: Error extracting references from dossier: {e}")
                 final_output = f"{edited_article}\n\n{references}"
                 llm_response.text = final_output
                 logger.info("Coordinator state: Final output assembled.")
                 step_completed = True
                 next_step = 99 # End successfully
        else:
             logger.warning(f"Coordinator state: Received unexpected text response during step {current_step}. Expected function call/response. Text: {llm_response.text[:100]}...")
             step_completed = False

    else:
        logger.warning(f"Coordinator state: LLM provided no actionable response for step {current_step}. Workflow may be stuck.")
        step_completed = False

    # Update State
    if next_step != current_step:
        state[COORD_STEP_KEY] = next_step
        logger.info(f"Coordinator state: Advanced to step {next_step}.")
    elif step_completed:
         logger.info(f"Coordinator state: Step {current_step} logic completed, but staying on step {current_step} (e.g., loop iteration).")
    else:
         logger.info(f"Coordinator state: Staying on step {current_step}.")


# --- Agent Definition ---

deep_research_coordinator = LlmAgent(
    name="DeepResearchCoordinatorAgent",
    model=agent_model,
    description="Orchestrates a deep research project using session state to track progress through planning, research, analysis, writing, and editing steps.",
    instruction=(
        "You are the Master Coordinator for a multi-agent deep research team. Your goal is to complete a 7-step workflow based on the user's initial request. You MUST use the session state (`COORD_STEP_KEY`) provided in the SYSTEM_INFO to determine which step to execute.\n\n"
        "**Workflow Steps (Execute based on `COORD_STEP_KEY`):**\n"
        "1.  **Call Planner:** If `COORD_STEP_KEY` is 1, call the `ResearchPlanner` agent using `transfer_to_agent` (provide *only* the agent name). It will use the initial query from history.\n"
        "2.  **Execute Research Tasks:** If `COORD_STEP_KEY` is 2, examine the `parsed_research_tasks` and `current_research_task_index` from state. Call the `ResearcherAgent` using `transfer_to_agent` (provide *only* the agent name) to execute the task at the current index. The task details will be passed implicitly. Repeat until all tasks are done.\n"
        "3.  **Aggregate Findings:** If `COORD_STEP_KEY` is 3, generate a text response simply stating 'Aggregating research findings.' (The callback will handle the aggregation).\n"
        "4.  **Call Analyzer:** If `COORD_STEP_KEY` is 4, call the `AnalysisLoopAgent` using `transfer_to_agent` (provide *only* the agent name). It will use the aggregated findings implicitly.\n"
        "5.  **Call Writer:** If `COORD_STEP_KEY` is 5, call the `WritingAgent` using `transfer_to_agent` (provide *only* the agent name). It will use the analysis dossier implicitly.\n"
        "6.  **Call Editor:** If `COORD_STEP_KEY` is 6, call the `EditorAgent` using `transfer_to_agent` (provide *only* the agent name). It will use the draft article and dossier implicitly.\n"
        "7.  **Assemble Output:** If `COORD_STEP_KEY` is 7, generate a text response simply stating 'Assembling final report.' (The callback will handle assembly).\n\n"
        "**Execution Rules:**\n"
        "-   **Check `COORD_STEP_KEY`:** Always determine your action based *only* on the current step number provided in the SYSTEM_INFO.\n"
        "-   **Sub-Agent Calls:** When calling sub-agents (`ResearchPlanner`, `ResearcherAgent`, `AnalysisLoopAgent`, `WritingAgent`, `EditorAgent`), use the `transfer_to_agent` function call and provide *only* the `agent_name` argument.\n"
        "-   **State Updates:** The system callbacks will handle updating the `COORD_STEP_KEY` and storing results. Focus only on executing the action for the current step.\n"
        "-   **Implicit Inputs:** Assume sub-agents automatically receive necessary data (like task details, findings, dossier) from the state/context when called.\n"
        "-   **Text for Processing Steps:** For steps 3 and 7, just output the simple text confirmation mentioned above.\n"
        "-   **Stick to the Plan:** Do not deviate from the 7-step workflow determined by the state."
    ),
    sub_agents=[
        research_planner_agent,
        researcher_agent,
        analysis_loop_agent,
        writing_agent,
        editor_agent,
    ],
    tools=[], # Coordinator doesn't call external tools directly
    before_model_callback=before_coord_model,
    after_model_callback=after_coord_model,
    # Corrected ToolConfig initialization
    tool_config=ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
        )
    )
)
