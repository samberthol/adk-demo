# agents/meta/agent.py
import os
import logging
from typing import Optional

# Corrected import for LlmAgent and CallbackContext
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
# Import Content/Part/FunctionCall for type checking in callback
# Import LlmResponse for the callback signature
from google.genai.types import Content, Part, FunctionCall
from google.adk.models import LlmResponse # <-- IMPORT LlmResponse

from agents.resource.agent import resource_agent
from agents.datascience.agent import data_science_agent
from agents.githubagent.agent import githubagent
from agents.mistral.agent import MistralVertexAgent # Import Mistral Agent

logger = logging.getLogger(__name__)

# Get the model for the MetaAgent itself
agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')

# Define the target agent name
MISTRAL_AGENT_NAME = "MistralChatAgent"

# --- Callback Function ---
def _save_input_for_mistral_agent(
    callback_context: CallbackContext, llm_response: LlmResponse # <-- CORRECT Type Hint
) -> None:
    """
    After model callback to save user input to state if transferring to Mistral.
    """
    transfer_to_mistral = False
    # Access parts via llm_response.content.parts
    if llm_response and llm_response.content and llm_response.content.parts: # <-- CORRECT Access
        part_content = llm_response.content.parts[0] # Get the first part
        func_calls = []
        # Handle cases where function_call might be single obj or list
        if hasattr(part_content, 'function_call'):
             call_data = getattr(part_content, 'function_call')
             if isinstance(call_data, FunctionCall):
                 func_calls = [call_data]
             elif isinstance(call_data, list):
                 func_calls = call_data

        for call in func_calls:
            if (
                isinstance(call, FunctionCall) and
                getattr(call, 'name', '') == 'transfer_to_agent' and
                isinstance(getattr(call, 'args', None), dict) and
                call.args.get('agent_name') == MISTRAL_AGENT_NAME
            ):
                transfer_to_mistral = True
                break

    if transfer_to_mistral:
        logger.info(f"[{callback_context.agent_name}] LLM decided to transfer to {MISTRAL_AGENT_NAME}. Saving preceding user input to state.")
        last_user_event = None
        history = callback_context.session.events or []
        for event in reversed(history):
            if event and event.author == 'user' and event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                last_user_event = event
                break

        if last_user_event:
            try:
                user_text = last_user_event.content.parts[0].text
                if user_text:
                    callback_context.add_state_delta({'mistral_input': user_text})
                    logger.info(f"[{callback_context.agent_name}] Saved input for {MISTRAL_AGENT_NAME}: '{user_text[:50]}...'")
                else:
                     logger.warning(f"[{callback_context.agent_name}] Preceding user event had no text content.")
            except Exception as e:
                 logger.error(f"[{callback_context.agent_name}] Failed to extract text from preceding user event: {e}", exc_info=True)
        else:
            logger.warning(f"[{callback_context.agent_name}] Could not find preceding user event in history to save for {MISTRAL_AGENT_NAME}.")

# --- Mistral Agent Instantiation ---
mistral_agent = None
try:
    mistral_agent = MistralVertexAgent(
        name=MISTRAL_AGENT_NAME,
        description="A conversational agent powered by Mistral via Vertex AI.",
        instruction="You are a helpful conversational AI assistant based on Mistral models."
    )
    logger.info(f"Successfully instantiated MistralVertexAgent ({MISTRAL_AGENT_NAME})")
except ValueError as e:
    logger.warning(f"Could not instantiate MistralVertexAgent - Missing Env Var(s): {e}")
except RuntimeError as e:
    logger.error(f"Could not instantiate MistralVertexAgent - SDK/Endpoint Init Error: {e}", exc_info=False)
except Exception as e:
    logger.error(f"Unexpected error instantiating MistralVertexAgent: {e}", exc_info=True)


# --- Build Active Sub-Agents List ---
active_sub_agents = [resource_agent, data_science_agent, githubagent]
if mistral_agent:
    active_sub_agents.append(mistral_agent)
else:
    logger.warning(f"{MISTRAL_AGENT_NAME} could not be initialized and will not be available.")


# --- Define Meta Agent (with callback) ---
meta_agent = LlmAgent(
    name="MetaAgent",
    model=agent_model,
    description="A helpful assistant coordinating specialized agents (resources, data, GitHub) and a general conversational agent (Mistral).",
    instruction=(
        "You are the primary assistant. Analyze the user's request.\n"
        "- If it involves managing cloud resources (like creating a VM or dataset), delegate the task to the 'ResourceAgent'.\n"
        "- If it involves querying data from BigQuery, delegate the task to the 'DataScienceAgent'.\n"
        "- If it involves searching GitHub or getting information from a GitHub repository, delegate the task to the 'githubagent'.\n"
        f"- If the request appears to be general conversation, requires summarization, explanation, brainstorming, or doesn't clearly fit other agents, delegate the task to the '{MISTRAL_AGENT_NAME}'.\n"
        "Clearly present the results from the specialist agents or the chat agent back to the user."
    ),
    sub_agents=active_sub_agents,
    after_model_callback=_save_input_for_mistral_agent,
)