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

    if current_step == 2:
        logger.info(f"Coordinator state (Before Step 2 LLM call): Current llm_request.contents (before adding context for step 2): {llm_request.contents}")

    step_info = WORKFLOW_STEPS.get(current_step)
    step_action_description = step_info['action'] if step_info else f"Unknown Step {current_step}"

    dynamic_context_text = ""
    appended_messages_count = 0

    if current_step == 2:
        current_task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
        parsed_tasks = state.get(PARSED_RESEARCH_TASKS_KEY, [])

        if parsed_tasks and 0 <= current_task_index < len(parsed_tasks):
            current_task_for_researcher = parsed_tasks[current_task_index]
            researcher_input_text = (
                f"RESEARCHER_TASK_INPUT:\n"
                f"You are the ResearcherAgent. Here is the specific task you need to execute:\n\n"
                f"{current_task_for_researcher}\n\n"
                f"Follow your detailed instructions for research execution and output format."
            )
            researcher_task_message = Content(role='user', parts=[Part(text=researcher_input_text)])
            llm_request.contents.append(researcher_task_message)
            appended_messages_count += 1
            logger.info(f"Coordinator state (Before Step 2 LLM call): Appended specific task for researcher (task index {current_task_index}): '{str(current_task_for_researcher)[:100]}...'")

            dynamic_context_text = (
                f"COORDINATOR_CONTEXT:\n"
                f"Current Step: {current_step} ({step_action_description})\n"
                f"Overall Research Goal (Initial User Query): '{initial_query[:200]}...'\n\n"
                f"Instruction: As the DeepResearchCoordinatorAgent, the specific research task for the ResearcherAgent (task {current_task_index + 1} of {len(parsed_tasks)}) has been prepared and added to the immediate context. "
                f"Your task is to now call 'transfer_to_agent' for the '{researcher_agent.name}' to execute this task. "
                f"Ensure you use the exact agent name '{researcher_agent.name}'."
            )
        else:
            logger.warning(
                f"Coordinator state (Step 2): Failed to retrieve valid research task. Index: {current_task_index}, Tasks available: {len(parsed_tasks)}. Instructing coordinator to handle error."
            )
            dynamic_context_text = (
                f"COORDINATOR_CONTEXT:\n"
                f"Current Step: {current_step} ({step_action_description})\n"
                f"Overall Research Goal (Initial User Query): '{initial_query[:200]}...'\n\n"
                f"Instruction: As the DeepResearchCoordinatorAgent, there was an error preparing the specific research task for the ResearcherAgent (task index {current_task_index} out of bounds or no tasks). "
                f"Please report this issue or try to replan if appropriate, according to your main instructions for handling errors."
            )
    else:
        # Standard instruction for other steps
        dynamic_context_text = (
            f"COORDINATOR_CONTEXT:\n"
            f"Current Step: {current_step} ({step_action_description})\n"
            f"Overall Research Goal (Initial User Query): '{initial_query[:200]}...'\n\n"
            f"Instruction: As the DeepResearchCoordinatorAgent, your task is to determine and request the correct next action for step {current_step}, which is '{step_action_description}'. "
            f"Consult your main agent instruction for how to perform this step (e.g., calling a specific sub-agent or preparing data)."
        )

    coordinator_context_message = Content(role='user', parts=[Part(text=dynamic_context_text)])
    llm_request.contents.append(coordinator_context_message)
    appended_messages_count += 1

    logger.info(f"Coordinator state: BEFORE model call. Current Step: {current_step}. Appended {appended_messages_count} message(s) to llm_request.contents.")


def after_coord_model(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    state = callback_context.state
    current_step_being_processed = state.get(COORD_STEP_KEY, 1) 
    step_info = WORKFLOW_STEPS.get(current_step_being_processed)

    logger.info(f"Coordinator state: AFTER model/sub-agent call. Processing for original step: {current_step_being_processed}. Raw llm_response: {llm_response}")

    if not step_info:
        logger.error(f"Coordinator state: Invalid step number {current_step_being_processed} found in state during 'after_coord_model'.")
        return

    next_step = current_step_being_processed
    step_completed_in_callback = False

    response_function_calls = getattr(llm_response, 'function_calls', None)
    response_function_responses = getattr(llm_response, 'function_responses', None)
    response_text = getattr(llm_response, 'text', None)

    is_coordinator_self_response = not response_function_responses # True if this is response from coordinator's own LLM call

    if is_coordinator_self_response:
        logger.info(f"Coordinator state (Processing Self LLM Response for Step {current_step_being_processed}):")
        if response_function_calls:
            logger.info(f"  - Decided to make function call(s): {response_function_calls}")
        elif response_text is not None:
            logger.info(f"  - Returned text: '{response_text[:200]}...'")
        else:
            logger.info("  - Returned no actionable output (no function calls, no text).")

    if response_function_calls:
        func_call = response_function_calls[0]
        logger.info(f"Coordinator state: LLM requested function call: {func_call.name}")
        expected_agent_name = step_info.get("agent")
        is_transfer = func_call.name == 'transfer_to_agent'
        func_args = getattr(func_call, 'args', {})

        if is_transfer and expected_agent_name and func_args.get('agent_name') == expected_agent_name:
            logger.info(f"Coordinator state: Correct agent transfer requested for step {current_step_being_processed}: {expected_agent_name}")
            if current_step_being_processed == 2:
                task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                logger.info(f"Coordinator state: Transferring to Researcher for task index {task_index}.")
        elif func_call.name != 'transfer_to_agent':
             logger.warning(f"Coordinator state: LLM called unexpected tool '{func_call.name}' during step {current_step_being_processed}.")
        else:
             logger.warning(f"Coordinator state: LLM requested transfer to agent '{func_args.get('agent_name')}' during step {current_step_being_processed}, but expected '{expected_agent_name}' or other mismatch.")

    elif response_function_responses:
        func_response = response_function_responses[0]
        func_response_name = getattr(func_response, 'name', 'UnknownFunction')
        raw_response_from_function = getattr(func_response, 'response', None)
        logger.info(f"Coordinator state: Received function response from sub-agent: {func_response_name} for step {current_step_being_processed}.")
        expected_agent_name = step_info.get("agent")

        if func_response_name == expected_agent_name:
            logger.info(f"Coordinator state: Result from expected sub-agent '{expected_agent_name}' for step {current_step_being_processed}.")
            response_data = raw_response_from_function if isinstance(raw_response_from_function, dict) else {}
            response_content = response_data.get("content", "Error: No content in response from sub-agent")
            logger.info(f"Coordinator state (Sub-agent {expected_agent_name} Response for Step {current_step_being_processed}): Raw response_content: '{str(response_content)[:500]}...'")

            if isinstance(response_content, str) and ("error" in response_content.lower() or "failed" in response_content.lower()):
                 logger.error(f"Coordinator state: Sub-agent {expected_agent_name} reported an error: {response_content}")
                 step_completed_in_callback = False
            else:
                if current_step_being_processed == 1: # Planner response
                    state[RESEARCH_PLAN_KEY] = response_content
                    logger.info("Coordinator state (Step 1 - Planner Response): Stored research plan.")
                    logger.info(f"Coordinator state (Step 1 - Planner Response): Raw plan content (first 500 chars): {str(response_content)[:500]}")
                    try:
                        tasks = []
                        current_task_lines = []
                        # Ensure response_content is not None and is a string before splitting
                        if response_content and isinstance(response_content, str):
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
                        else:
                            logger.error(f"Coordinator state (Step 1 - Planner Response): Planner response_content is None or not a string. Content: {response_content}")
                            # tasks will remain empty

                        if not tasks:
                             logger.error("Coordinator state (Step 1 - Planner Response): Failed to parse any valid tasks from the plan. Plan content was: " + str(response_content)[:300])
                             logger.info(f"Coordinator state (Step 1 - Planner Response): Task parsing resulted in {len(tasks)} tasks. step_completed_in_callback is being set to False.")
                             step_completed_in_callback = False
                        else:
                             state[PARSED_RESEARCH_TASKS_KEY] = tasks
                             state[CURRENT_RESEARCH_TASK_INDEX_KEY] = 0
                             logger.info(f"Coordinator state (Step 1 - Planner Response): Parsed {len(tasks)} tasks. Setting step_completed_in_callback = True.")
                             logger.info(f"Coordinator state (Step 1 - Planner Response): Successfully parsed {len(tasks)} tasks. step_completed_in_callback is being set to True.")
                             step_completed_in_callback = True
                    except Exception as e:
                         logger.error(f"Coordinator state (Step 1 - Planner Response): Error parsing research plan: {e}. Plan: {str(response_content)[:300]}", exc_info=True)
                         logger.info(f"Coordinator state (Step 1 - Planner Response): Exception during task parsing. step_completed_in_callback is being set to False.")
                         step_completed_in_callback = False
                elif current_step_being_processed == 2: # Researcher response
                    task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                    if RESEARCH_FINDINGS_KEY not in state: state[RESEARCH_FINDINGS_KEY] = []
                    current_findings = state.get(RESEARCH_FINDINGS_KEY)
                    if isinstance(current_findings, list):
                        current_findings.append(response_content)
                    else: # Should not happen if initialized correctly
                        logger.error(f"Coordinator state: {RESEARCH_FINDINGS_KEY} is not a list. Re-initializing.")
                        state[RESEARCH_FINDINGS_KEY] = [response_content]
                    logger.info(f"Coordinator state (Step 2 - Researcher Response): Stored finding for task index {task_index}.")
                    
                    next_task_index = task_index + 1
                    parsed_tasks = state.get(PARSED_RESEARCH_TASKS_KEY, [])
                    if next_task_index < len(parsed_tasks):
                        state[CURRENT_RESEARCH_TASK_INDEX_KEY] = next_task_index
                        logger.info(f"Coordinator state (Step 2 - Researcher Response): Advancing to next research task index {next_task_index}. step_completed_in_callback = False (to loop within step 2).")
                        step_completed_in_callback = False 
                    else:
                        logger.info("Coordinator state (Step 2 - Researcher Response): All research tasks completed. step_completed_in_callback = True.")
                        step_completed_in_callback = True # All tasks for step 2 are done
                elif current_step_being_processed == 4: # Analyzer response
                    state[ANALYSIS_DOSSIER_KEY] = response_content
                    logger.info("Coordinator state (Step 4 - Analyzer Response): Stored analysis dossier. step_completed_in_callback = True.")
                    step_completed_in_callback = True
                elif current_step_being_processed == 5: # Writer response
                    state[DRAFT_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state (Step 5 - Writer Response): Stored draft article. step_completed_in_callback = True.")
                    step_completed_in_callback = True
                elif current_step_being_processed == 6: # Editor response
                    state[EDITED_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state (Step 6 - Editor Response): Stored edited article. step_completed_in_callback = True.")
                    step_completed_in_callback = True
                else:
                     logger.warning(f"Coordinator state: Received function response for unhandled step {current_step_being_processed}")
                     step_completed_in_callback = False
                
                if step_completed_in_callback: # If sub-agent call for this step was successful and suggests step completion
                    if current_step_being_processed == 2 and not (next_task_index < len(parsed_tasks)): # Step 2 truly completes only if all tasks are done
                        next_step = current_step_being_processed + 1
                    elif current_step_being_processed != 2: # For other steps, if completed, advance
                        next_step = current_step_being_processed + 1
                    # If step 2 completed a task but not all tasks, next_step remains current_step_being_processed
                else: # if step_completed_in_callback is False
                    next_step = current_step_being_processed

                # NEW LOGGING FOR STEP 1 CONCLUSION - Placed after next_step is determined
                if current_step_being_processed == 1:
                    logger.info(f"Coordinator state (Step 1 Conclusion): current_step_being_processed={current_step_being_processed}, step_completed_in_callback={step_completed_in_callback}, determined next_step value={next_step}")
        else: # This 'else' corresponds to 'if func_response_name == expected_agent_name:'
             logger.warning(f"Coordinator state: Received function response from unexpected sub-agent '{func_response_name}'. Expected '{expected_agent_name}'.")

    elif response_text is not None: # Coordinator's own LLM returned text (not a function call/response)
        logger.info(f"Coordinator state: LLM provided direct text response for step {current_step_being_processed}.")
        if current_step_being_processed == 3: # Aggregate Findings
            logger.info("Coordinator state (Step 3 - Self Text Response): Executing 'Aggregate Findings' based on LLM text confirmation.")
            findings = state.get(RESEARCH_FINDINGS_KEY, [])
            if isinstance(findings, list) and findings:
                aggregated = "\n\n---\n\n".join(str(f) for f in findings)
                state[AGGREGATED_FINDINGS_KEY] = aggregated
                logger.info(f"Coordinator state (Step 3): Aggregated {len(findings)} findings. step_completed_in_callback = True.")
                step_completed_in_callback = True
                next_step = current_step_being_processed + 1
            else:
                logger.error(f"Coordinator state (Step 3): Cannot aggregate findings, none found or not a list. step_completed_in_callback = False.")
                step_completed_in_callback = False
        elif current_step_being_processed == 7: # Assemble Final Output
            logger.info("Coordinator state (Step 7 - Self Text Response): Executing 'Assemble Final Output' based on LLM text confirmation.")
            edited_article = state.get(EDITED_ARTICLE_KEY)
            analysis_dossier = state.get(ANALYSIS_DOSSIER_KEY)
            if not edited_article:
                 logger.error("Coordinator state (Step 7): Cannot assemble final output, edited article missing. step_completed_in_callback = False.")
                 if llm_response: setattr(llm_response, 'text', "Error: Failed to generate the final article as the edited version is missing.")
                 step_completed_in_callback = False
                 next_step = 99 
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
                                     next_section_marker = "\n## "
                                     end_of_ref_section = ref_content_block.find(next_section_marker)
                                     if end_of_ref_section != -1:
                                         ref_content_block = ref_content_block[:end_of_ref_section]
                                     references = f"## Sources Consulted\n{ref_content_block.strip()}"
                                else: 
                                    logger.warning("Coordinator state (Step 7): Found reference header but no content after it in dossier.")
                           else: 
                               logger.warning("Coordinator state (Step 7): 'Master Reference List' section header not found in dossier.")
                      except Exception as e: 
                           logger.error(f"Coordinator state (Step 7): Error extracting references from dossier: {e}", exc_info=True)
                 final_output = f"{edited_article}\n\n{references}"
                 if llm_response: setattr(llm_response, 'text', final_output)
                 logger.info("Coordinator state (Step 7): Final output assembled. step_completed_in_callback = True. Setting next_step = 99.")
                 step_completed_in_callback = True
                 next_step = 99 
        else:
             logger.warning(f"Coordinator state: Received unexpected text response during step {current_step_being_processed}. Text: {response_text[:100]}... step_completed_in_callback = False.")
             step_completed_in_callback = False
    else: # No function_calls, no function_responses, no text
        logger.warning(f"Coordinator state: LLM provided no actionable response for step {current_step_being_processed}. Workflow may be stuck. step_completed_in_callback = False.")
        step_completed_in_callback = False

    logger.info(f"Coordinator state (End of Step {current_step_being_processed} processing in callback): step_completed_in_callback = {step_completed_in_callback}, current_step_being_processed = {current_step_being_processed}, determined next_step = {next_step}")
    
    if next_step != current_step_being_processed:
        state[COORD_STEP_KEY] = next_step
        logger.info(f"Coordinator state: Advanced COORD_STEP_KEY in state to {next_step}.")
        if current_step_being_processed == 1 and next_step != current_step_being_processed : # Log if Step 1 successfully advanced
            logger.info(f"Coordinator state (Step 1 Conclusion): Successfully advanced COORD_STEP_KEY from 1 to {next_step}.")
    elif step_completed_in_callback and current_step_being_processed == 2 and (next_task_index < len(parsed_tasks)): # Special case for step 2 looping over tasks
        logger.info(f"Coordinator state: Step 2 task completed, but more tasks remain. Staying on COORD_STEP_KEY {current_step_being_processed} to process next task.")
    elif step_completed_in_callback:
         logger.info(f"Coordinator state: Step {current_step_being_processed} action processing completed according to callback logic, but next_step is still {next_step}. Waiting for next agent turn if applicable.")
    else:
         logger.info(f"Coordinator state: Staying on step {current_step_being_processed} for further processing or sub-agent response (step_completed_in_callback is False).")
    if current_step_being_processed == 1 and next_step == current_step_being_processed:
        logger.warning(f"Coordinator state (Step 1 Conclusion): COORD_STEP_KEY remains at {current_step_being_processed}. Agent will likely re-run Step 1 or stall. This usually indicates a problem in Step 1 processing if it was expected to advance.")


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