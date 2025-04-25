# agents/deepresearch/coordinator/agent.py
import os
import logging
from typing import Dict, Any, Optional, List, Tuple

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part

from ..planner.agent import research_planner_agent
from ..researcher.agent import researcher_agent
from ..loop.agent import analysis_loop_agent
from ..writer.agent import writing_agent
from ..editor.agent import editor_agent

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

WORKFLOW_STEPS: Dict[int, Dict[str, Any]] = {
    1: {"action": "call_planner", "agent": research_planner_agent.name},
    2: {"action": "execute_research_tasks", "agent": researcher_agent.name},
    3: {"action": "aggregate_findings"},
    4: {"action": "call_analyzer", "agent": analysis_loop_agent.name},
    5: {"action": "call_writer", "agent": writing_agent.name},
    6: {"action": "call_editor", "agent": editor_agent.name},
    7: {"action": "assemble_output"},
}

agent_model = 'gemini-2.5-pro-preview-03-25'
logger = logging.getLogger(__name__)

def before_coord_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    state = callback_context.state
    current_step = state.get(COORD_STEP_KEY, 1)
    initial_query = state.get(INITIAL_QUERY_KEY, "Initial query not captured yet.")

    if current_step == 1 and INITIAL_QUERY_KEY not in state:
        user_query_text = "User query not found in history."
        if llm_request.contents:
            for content_item in reversed(llm_request.contents):
                 if hasattr(content_item, 'role') and content_item.role == 'user' and hasattr(content_item, 'parts') and content_item.parts:
                     text_part_candidate = next((p.text for p in content_item.parts if hasattr(p, 'text')), None)
                     if text_part_candidate:
                         user_query_text = text_part_candidate
                         break
        state[INITIAL_QUERY_KEY] = user_query_text
        initial_query = user_query_text
        logger.info(f"Coordinator state: Stored initial query: '{user_query_text[:100]}...'")

    step_info = WORKFLOW_STEPS.get(current_step)
    step_action_description = step_info['action'] if step_info else f"Unknown Step {current_step}"
    
    dynamic_context_text = (
        f"COORDINATOR_CONTEXT:\n"
        f"Current Step: {current_step} ({step_action_description})\n"
        f"Overall Research Goal (Initial User Query): '{initial_query[:200]}...'\n\n"
        f"Instruction: As the DeepResearchCoordinatorAgent, your task is to determine and request the correct next action for step {current_step}, which is '{step_action_description}'. "
        f"Consult your main agent instruction for how to perform this step (e.g., calling a specific sub-agent or preparing data)."
    )
    
    context_as_user_message = Content(role='user', parts=[Part(text=dynamic_context_text)])
    llm_request.contents.append(context_as_user_message)

    logger.info(f"Coordinator state: BEFORE model call. Current Step: {current_step}. Appended dynamic context as 'user' message.")


def after_coord_model(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    state = callback_context.state
    current_step = state.get(COORD_STEP_KEY, 1)
    step_info = WORKFLOW_STEPS.get(current_step)

    logger.info(f"Coordinator state: AFTER model call. Current Step: {current_step}. Got LLM Response.")

    if not step_info:
        logger.error(f"Coordinator state: Invalid step number {current_step} found in state.")
        return

    next_step = current_step
    step_completed_in_callback = False

    # Safely access attributes using getattr
    response_function_calls = getattr(llm_response, 'function_calls', None)
    response_function_responses = getattr(llm_response, 'function_responses', None)
    response_text = getattr(llm_response, 'text', None)

    # Check the local variables that hold the result of getattr
    if response_function_calls:
        func_call = response_function_calls[0]
        logger.info(f"Coordinator state: LLM requested function call: {func_call.name}")
        expected_agent_name = step_info.get("agent")
        is_transfer = func_call.name == 'transfer_to_agent'
        # Ensure func_call.args exists before trying to access it
        func_args = getattr(func_call, 'args', {})

        if is_transfer and expected_agent_name and func_args.get('agent_name') == expected_agent_name:
            logger.info(f"Coordinator state: Correct agent transfer requested for step {current_step}: {expected_agent_name}")
            if current_step == 2:
                task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                logger.info(f"Coordinator state: Transferring to Researcher for task index {task_index}.")
        elif func_call.name != 'transfer_to_agent':
             logger.warning(f"Coordinator state: LLM called unexpected tool '{func_call.name}' during step {current_step}.")
        else:
             # This case implies is_transfer is True, but conditions for correct agent transfer failed
             logger.warning(f"Coordinator state: LLM requested transfer to agent '{func_args.get('agent_name')}' during step {current_step}, but expected '{expected_agent_name}' or other mismatch.")

    elif response_function_responses:
        func_response = response_function_responses[0]
        # Safely access func_response.name and func_response.response
        func_response_name = getattr(func_response, 'name', 'UnknownFunction')
        raw_response_from_function = getattr(func_response, 'response', None)

        logger.info(f"Coordinator state: Received function response for: {func_response_name}")
        expected_agent_name = step_info.get("agent")

        if func_response_name == expected_agent_name:
            logger.info(f"Coordinator state: Received result from expected agent '{expected_agent_name}' for step {current_step}.")
            
            response_data = raw_response_from_function if isinstance(raw_response_from_function, dict) else {}
            response_content = response_data.get("content", "Error: No content in response from sub-agent")


            if isinstance(response_content, str) and ("error" in response_content.lower() or "failed" in response_content.lower()):
                 logger.error(f"Coordinator state: Sub-agent {expected_agent_name} reported an error: {response_content}")
                 step_completed_in_callback = False
            else:
                if current_step == 1:
                    state[RESEARCH_PLAN_KEY] = response_content
                    logger.info("Coordinator state: Stored research plan.")
                    try:
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
                             step_completed_in_callback = False
                        else:
                             state[PARSED_RESEARCH_TASKS_KEY] = tasks
                             state[CURRENT_RESEARCH_TASK_INDEX_KEY] = 0
                             logger.info(f"Coordinator state: Parsed {len(tasks)} research tasks from plan titled '{plan_title}'.")
                             step_completed_in_callback = True
                    except Exception as e:
                         logger.error(f"Coordinator state: Error parsing research plan: {e}", exc_info=True)
                         step_completed_in_callback = False
                elif current_step == 2:
                    task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                    if RESEARCH_FINDINGS_KEY not in state: state[RESEARCH_FINDINGS_KEY] = []
                    
                    current_findings = state.get(RESEARCH_FINDINGS_KEY)
                    if isinstance(current_findings, list):
                        current_findings.append(response_content)
                        logger.info(f"Coordinator state: Stored finding for task index {task_index}.")
                    else:
                        logger.error(f"Coordinator state: {RESEARCH_FINDINGS_KEY} is not a list. Initializing and storing finding.")
                        state[RESEARCH_FINDINGS_KEY] = [response_content]
                    
                    next_task_index = task_index + 1
                    parsed_tasks = state.get(PARSED_RESEARCH_TASKS_KEY, [])
                    if next_task_index < len(parsed_tasks):
                        state[CURRENT_RESEARCH_TASK_INDEX_KEY] = next_task_index
                        logger.info(f"Coordinator state: Advancing to next research task index {next_task_index}.")
                        step_completed_in_callback = False 
                    else:
                        logger.info("Coordinator state: All research tasks completed.")
                        step_completed_in_callback = True
                elif current_step == 4:
                    state[ANALYSIS_DOSSIER_KEY] = response_content
                    logger.info("Coordinator state: Stored analysis dossier.")
                    step_completed_in_callback = True
                elif current_step == 5:
                    state[DRAFT_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state: Stored draft article.")
                    step_completed_in_callback = True
                elif current_step == 6:
                    state[EDITED_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state: Stored edited article.")
                    step_completed_in_callback = True
                else:
                     logger.warning(f"Coordinator state: Received function response for unexpected step {current_step}")
                     step_completed_in_callback = False
                
                if step_completed_in_callback and current_step != 2 : # Special handling for step 2 advancement
                    next_step = current_step + 1
                elif step_completed_in_callback and current_step == 2 and not (next_task_index < len(parsed_tasks)): # Check if it was the last task
                    next_step = current_step + 1

        else:
             logger.warning(f"Coordinator state: Received function response for unexpected function '{func_response_name}'. Expected '{expected_agent_name}'.")

    elif response_text is not None: # Check if response_text is not None, as empty string is also possible
        logger.info(f"Coordinator state: LLM provided direct text response for step {current_step}.")
        if current_step == 3:
            logger.info("Coordinator state: Executing Step 3 action: Aggregate Findings.")
            findings = state.get(RESEARCH_FINDINGS_KEY, [])
            if isinstance(findings, list) and findings:
                aggregated = "\n\n---\n\n".join(str(f) for f in findings)
                state[AGGREGATED_FINDINGS_KEY] = aggregated
                logger.info(f"Coordinator state: Aggregated {len(findings)} findings.")
                step_completed_in_callback = True
                next_step = current_step + 1
            else:
                logger.error(f"Coordinator state: Cannot aggregate findings, none found or not a list in state for step 3.")
                step_completed_in_callback = False
        elif current_step == 7:
            logger.info("Coordinator state: Executing Step 7 action: Assemble Final Output.")
            edited_article = state.get(EDITED_ARTICLE_KEY)
            analysis_dossier = state.get(ANALYSIS_DOSSIER_KEY)
            if not edited_article:
                 logger.error("Coordinator state: Cannot assemble final output, edited article missing for step 7.")
                 if llm_response: setattr(llm_response, 'text', "Error: Failed to generate the final article as the edited version is missing.")
                 step_completed_in_callback = False
                 next_step = 99 # Terminal error state
            else:
                 references = "## Sources Consulted\nNo sources were listed in the analysis dossier."
                 if analysis_dossier and isinstance(analysis_dossier, str):
                      try:
                           ref_header = "## VI. Master Reference List"
                           ref_start_index = analysis_dossier.find(ref_header)
                           if ref_start_index != -1:
                                content_start = analysis_dossier.find('\n', ref_start_index) + 1
                                if content_start > 0: # check if find returned -1 then +1 = 0
                                     ref_content_block = analysis_dossier[content_start:]
                                     # Try to find the start of the next H2 section to cap the reference list
                                     next_section_marker = "\n## "
                                     end_of_ref_section = ref_content_block.find(next_section_marker)
                                     if end_of_ref_section != -1:
                                         ref_content_block = ref_content_block[:end_of_ref_section]
                                     references = f"## Sources Consulted\n{ref_content_block.strip()}"
                                else: 
                                    logger.warning("Coordinator state: Found reference header but no content after it in dossier.")
                           else: 
                               logger.warning("Coordinator state: 'Master Reference List' section header not found in dossier.")
                      except Exception as e: 
                           logger.error(f"Coordinator state: Error extracting references from dossier: {e}", exc_info=True)
                 final_output = f"{edited_article}\n\n{references}"
                 if llm_response: setattr(llm_response, 'text', final_output)
                 logger.info("Coordinator state: Final output assembled for step 7.")
                 step_completed_in_callback = True
                 next_step = 99 # Terminal success state
        else:
             logger.warning(f"Coordinator state: Received unexpected text response during step {current_step}. Text: {response_text[:100]}...")
             step_completed_in_callback = False
    else:
        logger.warning(f"Coordinator state: LLM provided no actionable response (no function calls, no function responses, no text) for step {current_step}. Workflow may be stuck.")
        step_completed_in_callback = False

    if next_step != current_step:
        state[COORD_STEP_KEY] = next_step
        logger.info(f"Coordinator state: Advanced to step {next_step}.")
    elif step_completed_in_callback:
         logger.info(f"Coordinator state: Step {current_step} action processing completed. Staying on step {current_step} (or advancing internally within step logic like research tasks).")
    else:
         logger.info(f"Coordinator state: Staying on step {current_step} awaiting further action or sub-agent response.")

deep_research_coordinator = LlmAgent(
    name="DeepResearchCoordinatorAgent",
    model=agent_model,
    description="Orchestrates a deep research project using session state to track progress through planning, research, analysis, writing, and editing steps.",
    instruction=(
        "You are the Master Coordinator for a multi-agent deep research team. Your goal is to complete a 7-step workflow based on the user's initial request. "
        "The current step and context will be provided in the latest user message (COORDINATOR_CONTEXT).\n\n"
        "**Workflow Steps (Execute based on the current step from COORDINATOR_CONTEXT):**\n"
        "1.  **Call Planner:** If current step is 1, call the `ResearchPlanner` agent using `transfer_to_agent` (provide *only* the agent name). It will use the initial query information provided in the context.\n"
        "2.  **Execute Research Tasks:** If current step is 2, you will be orchestrating calls to the `ResearcherAgent`. Your callback logic handles state for `parsed_research_tasks` and `current_research_task_index`. Call the `ResearcherAgent` using `transfer_to_agent` (provide *only* the agent name) to execute the task for the current index. The callback will advance the index or step.\n"
        "3.  **Aggregate Findings:** If current step is 3, generate a text response simply stating 'Aggregating research findings.' (The system callback will handle the actual aggregation from state).\n"
        "4.  **Call Analyzer:** If current step is 4, call the `AnalysisLoopAgent` using `transfer_to_agent` (provide *only* the agent name). It will use aggregated findings from state.\n"
        "5.  **Call Writer:** If current step is 5, call the `WritingAgent` using `transfer_to_agent` (provide *only* the agent name). It will use the analysis dossier from state.\n"
        "6.  **Call Editor:** If current step is 6, call the `EditorAgent` using `transfer_to_agent` (provide *only* the agent name). It will use the draft article and dossier from state.\n"
        "7.  **Assemble Output:** If current step is 7, generate a text response simply stating 'Assembling final report.' (The system callback will handle actual assembly from state).\n\n"
        "**Execution Rules:**\n"
        "-   **Check COORDINATOR_CONTEXT:** Always determine your action based *only* on the current step number provided in the latest user message (COORDINATOR_CONTEXT).\n"
        "-   **Sub-Agent Calls:** When calling sub-agents, use the `transfer_to_agent` function call and specify *only* the target agent's name.\n"
        "-   **State Reliance:** Trust that system callbacks are managing state (like `COORD_STEP_KEY`, storing results, task indexing). Focus only on executing the action for the current step as described in COORDINATOR_CONTEXT.\n"
        "-   **Text for Processing Steps:** For steps 3 and 7, just output the simple text confirmation mentioned above. Your callback will do the work.\n"
        "-   **Stick to the Plan:** Do not deviate from the 7-step workflow. The COORDINATOR_CONTEXT is your primary guide for the current action."
    ),
    sub_agents=[
        research_planner_agent,
        researcher_agent,
        analysis_loop_agent,
        writing_agent,
        editor_agent,
    ],
    tools=[],
    before_model_callback=before_coord_model,
    after_model_callback=after_coord_model
)