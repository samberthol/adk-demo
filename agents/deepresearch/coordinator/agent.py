# agents/deepresearch/coordinator/agent.py
import os
import logging
from typing import Dict, Any, Optional, List, Tuple

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse, FunctionCall, Part
from google.adk.tools import ToolContext, FunctionResponse
from google.genai.types import Content, ToolConfig

from ..planner.agent import research_planner_agent
from ..researcher.agent import researcher_agent
from ..loop.agent import analysis_loop_agent
from ..writer.agent import writing_agent
from ..editor.agent import editor_agent

# --- State Keys ---
# These keys will be used in the session state dictionary
COORD_STEP_KEY = "coord_step"
RESEARCH_PLAN_KEY = "research_plan"
RESEARCH_FINDINGS_KEY = "research_findings" # List to store findings from each task
AGGREGATED_FINDINGS_KEY = "aggregated_findings"
ANALYSIS_DOSSIER_KEY = "analysis_dossier"
DRAFT_ARTICLE_KEY = "draft_article"
EDITED_ARTICLE_KEY = "edited_article"
CURRENT_RESEARCH_TASK_INDEX_KEY = "current_research_task_index" # To track progress within step 2
PARSED_RESEARCH_TASKS_KEY = "parsed_research_tasks" # Store the list of task blocks

# Define the sequence of steps and the sub-agent/action for each
# Step numbers are 1-based for clarity in instructions/state
WORKFLOW_STEPS: Dict[int, Dict[str, Any]] = {
    1: {"action": "call_planner", "agent": research_planner_agent.name},
    2: {"action": "execute_research_tasks", "agent": researcher_agent.name}, # This step involves a loop
    3: {"action": "aggregate_findings"}, # No agent call, just processing
    4: {"action": "call_analyzer", "agent": analysis_loop_agent.name},
    5: {"action": "call_writer", "agent": writing_agent.name},
    6: {"action": "call_editor", "agent": editor_agent.name},
    7: {"action": "assemble_output"}, # Final step, generates response
}
# --- End State Keys ---

agent_model = 'gemini-2.5-pro-preview-03-25'
logger = logging.getLogger(__name__)

# --- Callback Functions ---

def before_coord_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """
    Injects current step information into the LLM request.
    Reads state before the LLM call.
    """
    state = callback_context.session_context.state
    current_step = state.get(COORD_STEP_KEY, 1) # Default to step 1 if not set

    # Store the initial user query if it's the very first turn for this agent
    # This assumes the MetaAgent transferred control correctly.
    # The initial query should be the last 'user' message in history.
    if current_step == 1 and 'initial_query' not in state:
        user_query = "User query not found in history." # Default
        if llm_request.contents:
            # Look backwards for the last user message part
            for content in reversed(llm_request.contents):
                 if content.role == 'user' and content.parts:
                     # Find the first text part
                     text_part = next((p.text for p in content.parts if hasattr(p, 'text')), None)
                     if text_part:
                         user_query = text_part
                         break
        state['initial_query'] = user_query
        logger.info(f"Coordinator state: Stored initial query: '{user_query[:100]}...'")


    # Modify the prompt/instructions for the LLM based on the current step
    step_info = WORKFLOW_STEPS.get(current_step)
    step_description = f"Executing Step {current_step}: {step_info['action']}" if step_info else f"Unknown Step {current_step}"

    # Prepend state info to the last user message or add a new system message
    # (Adding a system message might be cleaner depending on ADK/model behavior)
    state_prompt = f"SYSTEM_INFO: You are currently on step {current_step} of the research workflow. Your goal is to execute this step based on the overall instructions and stored state. Step details: {step_description}.\n"

    # Find the last message to prepend to, or add as the first content item
    if llm_request.contents:
         # Try to find the last user or assistant message part to prepend to
         # This is a bit heuristic; ADK might offer better ways to inject context
         found = False
         for i in range(len(llm_request.contents) - 1, -1, -1):
             content = llm_request.contents[i]
             if content.role in ('user', 'assistant') and content.parts:
                 first_part = content.parts[0]
                 if hasattr(first_part, 'text'):
                      # Prepend state info to the existing text
                      original_text = first_part.text
                      llm_request.contents[i] = Content(role=content.role, parts=[Part(text=state_prompt + original_text)])
                      found = True
                      break
         if not found:
             # If no suitable message found, insert at the beginning
             llm_request.contents.insert(0, Content(role='system', parts=[Part(text=state_prompt[:-1])])) # Remove trailing newline if adding as system message
    else:
         # If no contents yet, add as a system instruction
         llm_request.contents = [Content(role='system', parts=[Part(text=state_prompt[:-1])])]


    logger.info(f"Coordinator state: BEFORE model call. Current Step: {current_step}. Prompt modified.")
    # logger.debug(f"Modified LLM Request Contents: {llm_request.contents}") # Optional: Log full modified prompt


def after_coord_model(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    """
    Processes the LLM response, updates state, and potentially advances the step.
    Writes state after the LLM call.
    """
    state = callback_context.session_context.state
    current_step = state.get(COORD_STEP_KEY, 1)
    step_info = WORKFLOW_STEPS.get(current_step)
    initial_query = state.get('initial_query', 'Missing Query') # Retrieve stored query

    logger.info(f"Coordinator state: AFTER model call. Current Step: {current_step}. Got LLM Response.")
    # logger.debug(f"LLM Response: {llm_response}") # Optional: Log full response

    if not step_info:
        logger.error(f"Coordinator state: Invalid step number {current_step} found in state.")
        # Potentially add error message to response or reset state
        return

    next_step = current_step # Default to staying on the same step unless advanced
    step_completed = False

    # --- Process based on current step ---

    # Check for function calls first (most common action)
    if llm_response.function_calls:
        func_call = llm_response.function_calls[0] # Assuming one call per turn for simplicity
        logger.info(f"Coordinator state: LLM requested function call: {func_call.name}")

        # Check if the function call matches the expected action for the current step
        expected_agent_name = step_info.get("agent")
        is_transfer = func_call.name == 'transfer_to_agent'

        if is_transfer and expected_agent_name and func_call.args.get('agent_name') == expected_agent_name:
            logger.info(f"Coordinator state: Correct agent transfer requested for step {current_step}: {expected_agent_name}")
            # State advancement will happen in the *next* turn's 'before_model'
            # after we receive the result from the sub-agent via FunctionResponse.
            # We store necessary context for the sub-agent if needed.
            if current_step == 1: # Calling Planner
                # Planner needs the initial query, which it gets from history. No extra state needed.
                pass
            elif current_step == 2: # Calling Researcher
                # Pass the specific task details.
                # This requires parsing the plan and tracking the current task index.
                task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                parsed_tasks = state.get(PARSED_RESEARCH_TASKS_KEY, [])
                if task_index < len(parsed_tasks):
                    # ADK should handle passing the primary argument (task block) implicitly
                    # when the LLM decides to call the researcher.
                    # We just log that the correct call is being made.
                     logger.info(f"Coordinator state: Transferring to Researcher for task index {task_index}.")
                else:
                     logger.error("Coordinator state: Tried to call researcher, but task index out of bounds.")
                     # Handle error - maybe reset step or add error message
            elif current_step == 4: # Calling Analyzer
                 # Pass aggregated findings. ADK handles implicit argument passing.
                 if AGGREGATED_FINDINGS_KEY not in state:
                      logger.error("Coordinator state: Tried to call analyzer, but aggregated findings missing.")
                 else:
                      logger.info("Coordinator state: Transferring to Analyzer.")
            elif current_step == 5: # Calling Writer
                 if ANALYSIS_DOSSIER_KEY not in state:
                      logger.error("Coordinator state: Tried to call writer, but analysis dossier missing.")
                 else:
                      logger.info("Coordinator state: Transferring to Writer.")
            elif current_step == 6: # Calling Editor
                 if DRAFT_ARTICLE_KEY not in state or ANALYSIS_DOSSIER_KEY not in state:
                      logger.error("Coordinator state: Tried to call editor, but draft article or dossier missing.")
                 else:
                      logger.info("Coordinator state: Transferring to Editor.")

            # We don't advance the step here; we wait for the FunctionResponse
            step_completed = False # Waiting for sub-agent result

        elif func_call.name != 'transfer_to_agent':
             logger.warning(f"Coordinator state: LLM called unexpected tool '{func_call.name}' during step {current_step}.")
             # Handle unexpected tool call if necessary
        else:
             # LLM called transfer_to_agent but to the wrong agent for the current step
             logger.warning(f"Coordinator state: LLM requested transfer to wrong agent '{func_call.args.get('agent_name')}' during step {current_step}. Expected '{expected_agent_name}'.")
             # Don't advance step, let LLM retry or handle error

    # Check for FunctionResponse (Result from a sub-agent call in the *previous* turn)
    elif llm_response.function_responses:
        func_response = llm_response.function_responses[0] # Assuming one response
        logger.info(f"Coordinator state: Received function response for: {func_response.name}")

        # Check if the response corresponds to the agent called in the previous turn's step
        # Note: We are currently in 'current_step', the response is for 'current_step - 1' action if it was an agent call.
        # This logic gets complex. Let's simplify: Assume the response is for the *just completed* action of the *current* step's agent.
        expected_agent_name = step_info.get("agent")

        # We need a way to map the response back to the step that initiated the call.
        # Let's assume the response relates to the agent expected for the *current* step.
        # This relies on the flow being strictly sequential turn-by-turn.

        if func_response.name == expected_agent_name:
            logger.info(f"Coordinator state: Received result from expected agent '{expected_agent_name}' for step {current_step}.")
            response_content = func_response.response.get("content", "Error: No content in response")

            if "error" in response_content.lower() or "failed" in response_content.lower():
                 logger.error(f"Coordinator state: Sub-agent {expected_agent_name} reported an error: {response_content}")
                 # Decide how to handle sub-agent errors (e.g., stop, retry, inform user)
                 # For now, we'll stop by not advancing the step.
                 step_completed = False
            else:
                # Store the result in the state based on the step
                if current_step == 1: # Planner result
                    state[RESEARCH_PLAN_KEY] = response_content
                    logger.info("Coordinator state: Stored research plan.")
                    # --- Parse the plan immediately for Step 2 ---
                    try:
                        # Simple parsing based on '---' separator and Task_ID presence
                        tasks = []
                        current_task = ""
                        lines = response_content.splitlines()
                        # Skip title line if present
                        start_index = 0
                        if lines and lines[0].startswith("#"):
                            start_index = 1
                        for line in lines[start_index:]:
                            if line.strip() == '---' and "**Task_ID:**" in current_task:
                                tasks.append(current_task.strip())
                                current_task = ""
                            else:
                                current_task += line + "\n"
                        if "**Task_ID:**" in current_task: # Add the last task
                             tasks.append(current_task.strip())

                        if not tasks:
                             logger.error("Coordinator state: Failed to parse any tasks from the plan.")
                             # Handle error - stop the process
                             step_completed = False
                        else:
                             state[PARSED_RESEARCH_TASKS_KEY] = tasks
                             state[CURRENT_RESEARCH_TASK_INDEX_KEY] = 0 # Initialize index for step 2
                             logger.info(f"Coordinator state: Parsed {len(tasks)} research tasks.")
                             step_completed = True # Ready to advance to step 2
                    except Exception as e:
                         logger.error(f"Coordinator state: Error parsing research plan: {e}", exc_info=True)
                         step_completed = False # Stop if parsing fails

                elif current_step == 2: # Researcher result for one task
                    task_index = state.get(CURRENT_RESEARCH_TASK_INDEX_KEY, 0)
                    if RESEARCH_FINDINGS_KEY not in state:
                        state[RESEARCH_FINDINGS_KEY] = []
                    state[RESEARCH_FINDINGS_KEY].append(response_content) # Append finding
                    logger.info(f"Coordinator state: Stored finding for task index {task_index}.")

                    # Check if more tasks remain in the plan
                    next_task_index = task_index + 1
                    parsed_tasks = state.get(PARSED_RESEARCH_TASKS_KEY, [])
                    if next_task_index < len(parsed_tasks):
                        # More tasks left, stay in step 2, update index
                        state[CURRENT_RESEARCH_TASK_INDEX_KEY] = next_task_index
                        logger.info(f"Coordinator state: Advancing to next research task index {next_task_index}.")
                        step_completed = False # Stay in step 2
                    else:
                        # All research tasks done, complete step 2
                        logger.info("Coordinator state: All research tasks completed.")
                        step_completed = True # Ready to advance to step 3

                elif current_step == 4: # Analyzer result
                    state[ANALYSIS_DOSSIER_KEY] = response_content
                    logger.info("Coordinator state: Stored analysis dossier.")
                    step_completed = True # Ready to advance to step 5
                elif current_step == 5: # Writer result
                    state[DRAFT_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state: Stored draft article.")
                    step_completed = True # Ready to advance to step 6
                elif current_step == 6: # Editor result
                    state[EDITED_ARTICLE_KEY] = response_content
                    logger.info("Coordinator state: Stored edited article.")
                    step_completed = True # Ready to advance to step 7
                else:
                     logger.warning(f"Coordinator state: Received function response for unexpected step {current_step}")
                     step_completed = False # Don't advance if logic is unclear

                # Advance step if the sub-step completed successfully
                if step_completed:
                    next_step = current_step + 1

        else:
             logger.warning(f"Coordinator state: Received function response for unexpected function '{func_response.name}'. Expected '{expected_agent_name}'.")
             # Don't advance step

    # Check for direct text response (for steps without agent calls or final output)
    elif llm_response.text:
        logger.info("Coordinator state: LLM provided direct text response.")
        if current_step == 3: # Aggregate Findings
            logger.info("Coordinator state: Executing Step 3: Aggregate Findings.")
            findings = state.get(RESEARCH_FINDINGS_KEY, [])
            if findings:
                aggregated = "\n\n---\n\n".join(findings)
                state[AGGREGATED_FINDINGS_KEY] = aggregated
                logger.info(f"Coordinator state: Aggregated {len(findings)} findings.")
                step_completed = True
                next_step = current_step + 1
            else:
                logger.error("Coordinator state: Cannot aggregate findings, none found in state.")
                step_completed = False # Stop if no findings
        elif current_step == 7: # Assemble Final Output
            logger.info("Coordinator state: Executing Step 7: Assemble Final Output.")
            edited_article = state.get(EDITED_ARTICLE_KEY)
            analysis_dossier = state.get(ANALYSIS_DOSSIER_KEY)

            if not edited_article:
                 logger.error("Coordinator state: Cannot assemble final output, edited article missing.")
                 # Potentially provide error message in response
                 llm_response.text = "Error: Failed to generate the final article."
                 step_completed = False # Mark as incomplete
                 next_step = 99 # Use a high number to indicate workflow end (error)
            else:
                 # Extract references from dossier
                 references = "## Sources Consulted\nNo sources were listed in the analysis dossier." # Default
                 if analysis_dossier:
                      try:
                           # Simple search for the reference section header
                           ref_start_index = analysis_dossier.find("## VI. Master Reference List")
                           if ref_start_index != -1:
                                ref_content = analysis_dossier[ref_start_index:]
                                # Find the end of the section (e.g., next H2 or end of string)
                                # This parsing is basic and might need refinement
                                next_h2 = ref_content.find("\n## ", len("## VI. Master Reference List"))
                                if next_h2 != -1:
                                     ref_content = ref_content[:next_h2]
                                # Clean up the header maybe
                                ref_content = ref_content.replace("## VI. Master Reference List", "## Sources Consulted", 1).strip()
                                references = ref_content
                           else:
                               logger.warning("Coordinator state: 'Master Reference List' section not found in dossier.")
                      except Exception as e:
                           logger.error(f"Coordinator state: Error extracting references from dossier: {e}")

                 final_output = f"{edited_article}\n\n{references}"
                 # Modify the LLMResponse to contain the final assembled output
                 # This assumes the LLM didn't already generate it perfectly.
                 llm_response.text = final_output
                 logger.info("Coordinator state: Final output assembled.")
                 step_completed = True
                 next_step = 99 # Use a high number to indicate workflow end (success)
        else:
             logger.warning(f"Coordinator state: Received unexpected text response during step {current_step}.")
             # Don't advance step

    else:
        # No function calls, no function responses, no text -> LLM likely halted or gave empty response
        logger.warning(f"Coordinator state: LLM provided no actionable response for step {current_step}. Halting workflow here.")
        # Don't advance the step, let it retry on the next turn or indicate failure.
        step_completed = False


    # --- Update State ---
    if next_step != current_step:
        state[COORD_STEP_KEY] = next_step
        logger.info(f"Coordinator state: Advanced to step {next_step}.")
    elif step_completed:
         # This case shouldn't happen if logic is correct (completed should imply next_step > current_step)
         logger.warning(f"Coordinator state: Step {current_step} marked completed, but next_step did not advance.")
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
    # Enable function calling for agent transfers
    tool_config=ToolConfig(function_calling_config="auto")
)
