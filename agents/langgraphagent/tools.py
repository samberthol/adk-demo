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

LANGGRAPH_A2A_ENDPOINT = os.environ.get("LANGGRAPH_A2A_ENDPOINT")

def _send_a2a_json_rpc_request(method: str, params: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    if not LANGGRAPH_A2A_ENDPOINT:
        raise ValueError("LANGGRAPH_A2A_ENDPOINT environment variable is not set.")

    request_id = str(uuid.uuid4())

    if session_id:
        params['sessionId'] = session_id
    elif 'sessionId' not in params:
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
            timeout=60
        )
        response.raise_for_status()
        response_data = response.json()

        if response_data.get("id") != request_id:
             logger.warning(f"JSON-RPC response ID mismatch. Expected {request_id}, got {response_data.get('id')}")

        if "error" in response_data:
             logger.error(f"JSON-RPC error response: {response_data['error']}")
             error_details = response_data['error'].get('message', 'Unknown error')
             error_code = response_data['error'].get('code', 'N/A')
             raise RuntimeError(f"LangGraph agent returned error (Code: {error_code}): {error_details}")
        elif "result" not in response_data:
             logger.error(f"Invalid JSON-RPC response: missing 'result' and 'error'. Response: {response_data}")
             raise RuntimeError("Invalid JSON-RPC response from LangGraph agent (missing 'result' or 'error').")

        logger.debug(f"Received JSON-RPC response ID={response_data.get('id')}, Session={response_data.get('result', {}).get('sessionId')}")
        return response_data

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
        logger.error(f"Unexpected error during A2A communication: {e}", exc_info=True)
        raise


def langgraph_currency_a2a_tool_func(query: str, session_id: Optional[str] = None) -> dict:
    task_id = f"adk-task-{uuid.uuid4()}"
    current_session_id = session_id

    try:
        logger.info(f"Sending A2A task to LangGraph Currency Agent. TaskID: {task_id}, SessionID: {current_session_id}, Query: '{query}'")
        send_params = {
            "id": task_id,
            "acceptedOutputModes": ["text"],
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": query}]
            }
        }

        response_data = _send_a2a_json_rpc_request("tasks/send", send_params, current_session_id)

        result_payload = response_data.get("result", {})
        returned_session_id = result_payload.get("sessionId", current_session_id or send_params.get('sessionId'))
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
                "status": "success",
                "result": prompt_for_input,
                "session_id": returned_session_id,
                "state": current_state
            }

        elif current_state in ["failed", "cancelled", "error"]:
             error_message = task_status.get("message", {}).get("parts", [{}])[0].get("text", f"Task ended with unhandled state: {current_state}")
             logger.error(f"Task {task_id} ended with state: {current_state}. Message: {error_message}. Response: {response_data}")
             return {
                 "status": "error",
                 "error": f"LangGraph agent task {current_state}: {error_message}",
                 "session_id": returned_session_id,
                 "state": current_state
            }
        else:
            logger.error(f"Task {task_id} returned unexpected state: {current_state}. Response: {response_data}")
            return {
                "status": "error",
                "error": f"LangGraph agent returned unexpected state: {current_state}",
                "session_id": returned_session_id,
                "state": current_state
            }

    except ValueError as e:
         logger.error(f"Configuration Error: {e}", exc_info=True)
         return {"status": "error", "error": f"Configuration Error: {e}", "session_id": current_session_id, "state": "error"}
    except TimeoutError as e:
        logger.error(f"Timeout Error: {e}", exc_info=True)
        return {"status": "error", "error": f"Communication Error: {e}", "session_id": current_session_id, "state": "error"}
    except RuntimeError as e:
         logger.error(f"Runtime Error: {e}", exc_info=True)
         return {"status": "error", "error": f"Communication Error: {e}", "session_id": current_session_id, "state": "error"}
    except Exception as e:
        logger.exception(f"An unexpected error occurred in langgraph_currency_a2a_tool_func:")
        return {"status": "error", "error": f"An unexpected error occurred: {str(e)}", "session_id": current_session_id, "state": "error"}


langgraph_currency_tool = FunctionTool(func=langgraph_currency_a2a_tool_func)