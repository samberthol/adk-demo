# adk-demo/agents/githubagent/tools.py
import os
import requests
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") # May be needed by MCP server or direct calls

# --- Helper Function to Call MCP Server (via mcpo REST API) ---
def _invoke_mcp_tool(tool_name: str, inputs: dict) -> str:
    """
    Helper function to invoke a specific tool on the MCP server wrapped by mcpo.
    Constructs the URL based on the tool name. Includes enhanced logging.
    """
    if not MCP_SERVER_URL:
        logger.error("MCP_SERVER_URL is not set in environment variables.")
        return "Error: MCP_SERVER_URL environment variable is not set. Cannot contact MCP server."

    tool_invoke_url = f"{MCP_SERVER_URL.rstrip('/')}/{tool_name}"

    headers = {
        "Content-Type": "application/json",
        # Add API Key header if mcpo was started with --api-key
        # "Authorization": f"Bearer {YOUR_MCPO_API_KEY}"
    }
    payload = inputs

    logger.info(f"Attempting to invoke MCP tool via mcpo:")
    logger.info(f"  URL: POST {tool_invoke_url}")
    logger.info(f"  Headers: {headers}")
    logger.info(f"  Payload: {json.dumps(payload)}") # Log the payload as JSON string

    try:
        response = requests.post(tool_invoke_url, headers=headers, json=payload, timeout=60)

        # --- Enhanced Logging ---
        logger.info(f"Received response from mcpo:")
        logger.info(f"  Status Code: {response.status_code}")
        # Log first 500 chars of response text for brevity, handle potential decoding errors
        try:
            response_text_preview = response.text[:500]
            logger.info(f"  Response Text Preview: {response_text_preview}{'...' if len(response.text) > 500 else ''}")
        except Exception as log_e:
             logger.warning(f"Could not log response text preview: {log_e}")
        # --- End Enhanced Logging ---

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()
        if isinstance(response_data, dict):
             return json.dumps(response_data, indent=2)
        elif isinstance(response_data, str):
             return response_data
        else:
             return str(response_data)

    except requests.exceptions.RequestException as e:
        # Log details even before returning the error string
        logger.error(f"HTTP Request Error calling mcpo wrapper for tool '{tool_name}': {e}", exc_info=True)
        error_status = "N/A"
        error_text = str(e)
        if e.response is not None:
             error_status = e.response.status_code
             try:
                  error_text = e.response.text
             except Exception:
                  pass # Keep original error string if text cannot be read
             logger.error(f"  Error Status Code: {error_status}")
             logger.error(f"  Error Response Text: {error_text}")
        # --- Make error message more informative ---
        return f"Error communicating with the MCP server: Status {error_status} - Response: {error_text}"
        # --- Old error message handling ---
        # error_details = str(e)
        # if e.response is not None:
        #      try:
        #           error_details = f"Status {e.response.status_code}: {e.response.text}"
        #      except Exception:
        #           pass
        # if e.response is not None and e.response.status_code == 404:
        #      return f"Error: The tool endpoint '{tool_invoke_url}' was not found on the MCP server. Check if the tool name '{tool_name}' is correct and exposed by the server."
        # return f"Error communicating with the MCP server via mcpo: {error_details}"
    except json.JSONDecodeError:
        # Log the response text that failed to parse
        logger.error(f"Failed to decode JSON response from mcpo for tool '{tool_name}'. Status: {response.status_code}. Response Text: {response.text}")
        return f"Error: Received invalid JSON response from mcpo wrapper (Status: {response.status_code}). Check agent logs for response text."
    except Exception as e:
        logger.error(f"Unexpected error invoking mcpo-wrapped MCP tool '{tool_name}': {e}", exc_info=True)
        return f"An unexpected error occurred while calling the MCP tool via mcpo: {str(e)}"


# --- GitHub Agent Tools (using MCP via mcpo) ---
# Functions search_github_repositories_func and get_github_repo_file_func remain the same as before
# ... (keep the existing function definitions for search_github_repositories_func and get_github_repo_file_func) ...

def search_github_repositories_func(query: str = "") -> str:
    """
    Searches GitHub repositories using the 'search_repositories' tool via the MCP server (mcpo wrapper).
    Requires a search query.
    """
    if not query:
        return "GitHub search failed: Search query is required."
    mcp_tool_name = "search_repositories"
    inputs = {"query": query}
    return _invoke_mcp_tool(mcp_tool_name, inputs)

def get_github_repo_file_func(owner: str = "", repo: str = "", path: str = "") -> str:
    """
    Gets the content of a file from a GitHub repository using the 'get_file_contents' tool via the MCP server (mcpo wrapper).
    Requires repository owner, repository name, and the file path (e.g., 'README.md').
    """
    if not owner: return "Get file failed: Repository owner is required."
    if not repo: return "Get file failed: Repository name is required."
    if not path: return "Get file failed: File path is required."
    mcp_tool_name = "get_file_contents"
    inputs = { "owner": owner, "repo": repo, "path": path }
    return _invoke_mcp_tool(mcp_tool_name, inputs)