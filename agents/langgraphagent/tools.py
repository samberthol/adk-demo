# adk-demo/agents/langgraphagent/tools.py
import os
import requests
import json
import logging
import time
import uuid
from google.adk.tools import FunctionTool
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configuration: Get the LangGraph Currency agent's A2A JSON-RPC endpoint
# This will point to the separate Cloud Run service where the LangGraph agent is deployed.
LANGGRAPH_A2A_ENDPOINT = os.environ.get("LANGGRAPH_A2A_ENDPOINT")
# Example: export LANGGRAPH_A2A_ENDPOINT="http://langgraph-currency-agent-service.a.run.app" # Placeholder URL

# Configuration for polling (if needed, though tasks/send seems synchronous in the example)
# POLLING_INTERVAL_SECONDS = 2
# MAX_POLLING_ATTEMPTS = 15

# --- Helper function to send JSON-RPC requests ---
def _send_a2a_json_rpc_request(method: str, params: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Helper function to send a JSON-RPC request to the LangGraph Currency Agent.
    Includes session ID management.
    """
    if not LANGGRAPH_A2A_ENDPOINT:
        raise ValueError("LANGGRAPH_A2A_ENDPOINT environment variable is not set.")

    request_id = str(uuid.uuid4()) # Unique ID for this specific request

    # Ensure session ID is included if provided
    if session_id:
        params['sessionId'] = session_id
    elif 'sessionId' not in params:
        # Generate a new session ID if none is provided and not already in params
        # Note: For multi-turn, the caller (agent) should manage and reuse the session ID.
        params['sessionId'] = str(uuid.uuid4())
        logger.warning(f"No session ID provided for method {method}, generated a new one: {params['sessionId']}")


    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": request_id
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    logger.debug(f"Sending JSON-RPC request to {LANGGRAPH_A2A_ENDPOINT}: Method={method}, ID={request_id}, Session={params.get('sessionId')}")
    logger.debug(f"Payload: {json.dumps(payload)}")

    try:
        response = requests.post(
            LANGGRAPH_A2A_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60 # Increased timeout for potentially longer agent processing
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()

        # Basic validation of the response structure
        if response_data.get("id") != request_id:
             logger.warning(f"JSON-RPC response ID mismatch. Expected {request_id}, got {response_data.get('id')}")

        if "error" in response_data:
             logger.error(f"JSON-RPC error response: {response_data['error']}")
             # Propagate the error details if possible
             error_details = response_data['error'].get('message', 'Unknown error')
             error_code = response_data['error'].get('code', 'N/A')
             raise RuntimeError(f"LangGraph agent returned error (Code: {error_code}): {error_details}")
        elif "result" not in response_data:
             logger.error(f"Invalid JSON-RPC response: missing 'result' and 'error'. Response: {response_data}")
             raise RuntimeError("Invalid JSON-RPC response from LangGraph agent (missing 'result' or 'error').")

        logger.debug(f"Received JSON-RPC response ID={response_data.get('id')}, Session={response_data.get('result', {}).get('sessionId')}")
        return response_data # Return the full response dictionary

    except requests.exceptions.Timeout:
        logger.error(f"Timeout during JSON-RPC request to {LANGGRAPH_A2A_ENDPOINT} for method {method}")
        raise TimeoutError(f"Request to LangGraph agent timed out.")
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Error during JSON-RPC request to {LANGGRAPH_A2A_ENDPOINT}: {e}")
        raise RuntimeError(f"Failed to communicate with the LangGraph agent: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from {LANGGRAPH_A2A_ENDPOINT}: {e}. Response text: {response.text}")
        raise RuntimeError(f"Invalid JSON response received from LangGraph agent.") from e
    except Exception as e:
        # Catch potential RuntimeErrors raised above or other unexpected errors
        logger.error(f"Unexpected error during A2A communication: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the calling agent


# --- Tool logic defined as a function ---
def langgraph_currency_a2a_tool_func(query: str, session_id: Optional[str] = None) -> dict:
    """
    Sends a query as a task to the external LangGraph Currency Agent using the
    A2A JSON-RPC protocol (method 'tasks/send') and returns its response.
    Manages session ID for potential multi-turn conversations.

    Args:
        query: The natural language query for the LangGraph Currency agent.
        session_id: Optional existing session ID to continue a conversation.
                    If None, a new session might be initiated by the helper.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'
        - 'result': The text response from the agent (if successful).
        - 'error': An error message (if failed).
        - 'session_id': The session ID used/returned by the agent (important for multi-turn).
        - 'state': The final state reported by the agent ('completed', 'input-required', etc.)
    """
    task_id = f"adk-task-{uuid.uuid4()}" # Generate a unique task ID for this interaction
    current_session_id = session_id # Use provided session ID or None

    try:
        # --- Use tasks/send method ---
        logger.info(f"Sending A2A task to LangGraph Currency Agent. TaskID: {task_id}, SessionID: {current_session_id}, Query: '{query}'")
        send_params = {
            "id": task_id,
            # session_id will be added by _send_a2a_json_rpc_request if current_session_id is set
            "acceptedOutputModes": ["text"], # Assuming text output
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": query}]
            }
        }

        # Pass the current_session_id to the helper
        response_data = _send_a2a_json_rpc_request("tasks/send", send_params, current_session_id)

        # Process the result from tasks/send
        result_payload = response_data.get("result", {})
        returned_session_id = result_payload.get("sessionId", current_session_id or send_params.get('sessionId')) # Get session ID back
        task_status = result_payload.get("status", {})
        current_state = task_status.get("state")

        logger.info(f"A2A Task {task_id} response received. State: {current_state}, SessionID: {returned_session_id}")

        if current_state == "completed":
            artifacts = result_payload.get("artifacts", [])
            if artifacts and artifacts[0].get("parts"):
                 parts = artifacts[0]["parts"]
                 for part in parts:
                      if part.get("type") == "text":
                           final_response = part.get("text", "Error: Agent returned empty text part.")
                           logger.info(f"Extracted 'completed' response for task {task_id}.")
                           return {
                               "status": "success",
                               "result": final_response,
                               "session_id": returned_session_id,
                               "state": current_state
                           }
            logger.warning(f"Task {task_id} completed but no suitable text artifact found. Response: {response_data}")
            return {
                "status": "error",
                "error": "LangGraph agent completed the task but provided no text response.",
                "session_id": returned_session_id,
                "state": current_state
            }

        elif current_state == "input-required":
            input_message = task_status.get("message", {})
            parts = input_message.get("parts", [])
            prompt_for_input = "Agent requires more input, but did not provide specific text."
            for part in parts:
                if part.get("type") == "text":
                    prompt_for_input = part.get("text")
                    break
            logger.info(f"Task {task_id} requires input. Prompt: '{prompt_for_input}'")
            return {
                "status": "success", # Still success, but requires follow-up
                "result": prompt_for_input, # Return the agent's question
                "session_id": returned_session_id,
                "state": current_state # Indicate input is needed
            }

        elif current_state in ["failed", "cancelled", "error"]: # Handle other terminal states
             error_message = task_status.get("message", {}).get("parts", [{}])[0].get("text", f"Task ended with unhandled state: {current_state}")
             logger.error(f"Task {task_id} ended with state: {current_state}. Message: {error_message}. Response: {response_data}")
             return {
                 "status": "error",
                 "error": f"LangGraph agent task {current_state}: {error_message}",
                 "session_id": returned_session_id,
                 "state": current_state
            }
        else: # Handle unexpected states
            logger.error(f"Task {task_id} returned unexpected state: {current_state}. Response: {response_data}")
            return {
                "status": "error",
                "error": f"LangGraph agent returned unexpected state: {current_state}",
                "session_id": returned_session_id,
                "state": current_state
            }

    except ValueError as e: # Catch config errors (e.g., missing endpoint)
         logger.error(f"Configuration Error: {e}", exc_info=True)
         return {"status": "error", "error": f"Configuration Error: {e}", "session_id": current_session_id, "state": "error"}
    except TimeoutError as e:
        logger.error(f"Timeout Error: {e}", exc_info=True)
        return {"status": "error", "error": f"Communication Error: {e}", "session_id": current_session_id, "state": "error"}
    except RuntimeError as e: # Catch communication or JSON-RPC errors from helper
         logger.error(f"Runtime Error: {e}", exc_info=True)
         return {"status": "error", "error": f"Communication Error: {e}", "session_id": current_session_id, "state": "error"}
    except Exception as e:
        logger.exception(f"An unexpected error occurred in langgraph_currency_a2a_tool_func:")
        return {"status": "error", "error": f"An unexpected error occurred: {str(e)}", "session_id": current_session_id, "state": "error"}


# --- Instantiate the tool using FunctionTool ---
# The name and description are derived from the function definition.
# We explicitly pass the function to the 'func' argument.
langgraph_currency_tool = FunctionTool(func=langgraph_currency_a2a_tool_func)
