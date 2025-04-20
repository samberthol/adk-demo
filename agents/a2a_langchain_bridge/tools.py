# adk-demo/agents/a2a_langchain_bridge/tools.py
import os
import requests
import json
import logging
import time
import uuid
from google.adk.tools import FunctionTool
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Configuration: Get the LangChain agent's A2A JSON-RPC endpoint
LANGCHAIN_A2A_ENDPOINT = os.environ.get("LANGCHAIN_A2A_ENDPOINT")
# Example: export LANGCHAIN_A2A_ENDPOINT="http://localhost:8080/a2a" # Default for the sample

# Configuration for polling getTask
POLLING_INTERVAL_SECONDS = 2  # How often to check task status
MAX_POLLING_ATTEMPTS = 15     # Max number of times to poll (~30 seconds total)

# --- Helper function ---
def _send_a2a_json_rpc_request(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to send a JSON-RPC request."""
    if not LANGCHAIN_A2A_ENDPOINT:
        raise ValueError("LANGCHAIN_A2A_ENDPOINT environment variable is not set.")

    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": request_id
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    logger.debug(f"Sending JSON-RPC request to {LANGCHAIN_A2A_ENDPOINT}: Method={method}, ID={request_id}")
    logger.debug(f"Payload: {json.dumps(payload)}")

    try:
        response = requests.post(
            LANGCHAIN_A2A_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        response_data = response.json()

        if response_data.get("id") != request_id:
             logger.warning(f"JSON-RPC response ID mismatch. Expected {request_id}, got {response_data.get('id')}")

        if "error" in response_data:
             logger.error(f"JSON-RPC error response: {response_data['error']}")
             raise RuntimeError(f"LangGraph agent returned error: {response_data['error'].get('message', 'Unknown error')}")
        elif "result" not in response_data:
             logger.error(f"Invalid JSON-RPC response: missing 'result' and 'error'. Response: {response_data}")
             raise RuntimeError("Invalid JSON-RPC response from LangGraph agent.")

        logger.debug(f"Received JSON-RPC response ID={response_data.get('id')}")
        return response_data

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Error during JSON-RPC request to {LANGCHAIN_A2A_ENDPOINT}: {e}")
        raise RuntimeError(f"Failed to communicate with the LangGraph agent: {e}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from {LANGCHAIN_A2A_ENDPOINT}: {e}. Response text: {response.text}")
        raise RuntimeError(f"Invalid JSON response received from LangGraph agent.") from e

# --- Tool logic defined as a function with a descriptive name ---
def langchain_a2a_json_rpc_client(query: str) -> dict:
    """
    Sends a query as a task to a configured external LangGraph agent using the
    A2A JSON-RPC protocol (createTask/getTask) and returns its response.

    Args:
        query: The natural language query for the LangGraph agent.

    Returns:
        A dictionary containing the response text from the LangGraph agent
        under the 'result' key ('status': 'success'), or an error message under the 'error' key ('status': 'error').
    """
    try:
        # --- 1. Create Task ---
        logger.info(f"Creating A2A task for query: '{query}'")
        create_params = {
            "task": {
                "messages": [
                    {
                        "role": "user",
                        "parts": [{"type": "text", "text": query}]
                    }
                ]
            }
        }
        create_response = _send_a2a_json_rpc_request("createTask", create_params)
        task_id = create_response.get("result", {}).get("taskId")

        if not task_id:
            logger.error(f"createTask response did not contain taskId. Response: {create_response}")
            return {"status": "error", "error": "Failed to create task on LangGraph agent (missing taskId)."}
        logger.info(f"A2A Task created successfully. taskId: {task_id}")

        # --- 2. Poll Get Task ---
        logger.info(f"Polling getTask for taskId {task_id}...")
        for attempt in range(MAX_POLLING_ATTEMPTS):
            logger.debug(f"Polling attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS} for taskId {task_id}")
            time.sleep(POLLING_INTERVAL_SECONDS)

            get_params = {"taskId": task_id}
            get_response = _send_a2a_json_rpc_request("getTask", get_params)
            task_result = get_response.get("result", {})
            task_data = task_result.get("task", {})
            task_state = task_data.get("state")
            logger.debug(f"Task {task_id} state: {task_state}")

            if task_state == "completed":
                logger.info(f"Task {task_id} completed.")
                artifacts = task_data.get("artifacts", [])
                if artifacts and artifacts[0].get("messages"):
                     parts = artifacts[0]["messages"][0].get("parts", [])
                     for part in parts:
                          if part.get("type") == "text":
                               final_response = part.get("text", "Error: Agent returned empty text part.")
                               logger.info(f"Extracted response for task {task_id}.")
                               return {"status": "success", "result": final_response} # Return dict
                logger.warning(f"Task {task_id} completed but no suitable text artifact found. Response: {get_response}")
                return {"status": "error", "error": "LangGraph agent completed the task but provided no text response."}

            elif task_state in ["failed", "cancelled"]:
                 logger.error(f"Task {task_id} ended with state: {task_state}. Response: {get_response}")
                 error_message = task_data.get("status", {}).get("errorMessage", f"Task failed or was cancelled.")
                 return {"status": "error", "error": f"LangGraph agent task {task_state}: {error_message}"}

        logger.warning(f"Task {task_id} did not complete within the polling timeout.")
        return {"status": "error", "error": "The request to the LangGraph agent timed out waiting for task completion."}

    except ValueError as e:
         return {"status": "error", "error": f"Configuration Error: {e}"}
    except RuntimeError as e:
         return {"status": "error", "error": f"Communication Error: {e}"}
    except Exception as e:
        logger.exception(f"An unexpected error occurred in langchain_a2a_json_rpc_client:")
        return {"status": "error", "error": f"An unexpected error occurred during A2A communication: {str(e)}"}


# --- Instantiate the tool using FunctionTool (Corrected: only func argument) ---
# The name and description are derived from the function definition above.
langchain_a2a_tool = FunctionTool(func=langchain_a2a_json_rpc_client)