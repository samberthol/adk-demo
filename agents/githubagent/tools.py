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
    Constructs the URL based on the tool name.
    """
    if not MCP_SERVER_URL:
        return "Error: MCP_SERVER_URL environment variable is not set. Cannot contact MCP server."

    # Construct the URL dynamically based on the tool name (mcpo convention)
    # Example: http://host:port/search_repositories
    tool_invoke_url = f"{MCP_SERVER_URL.rstrip('/')}/{tool_name}" # Ensure no double slashes

    headers = {
        "Content-Type": "application/json",
        # Add API Key header if mcpo was started with --api-key
        # "Authorization": f"Bearer {YOUR_MCPO_API_KEY}"
        # Note: GITHUB_TOKEN is likely used by the MCP server internally, not needed here unless mcpo requires it.
    }
    # mcpo expects the inputs directly as the JSON body for the POST request
    payload = inputs

    logger.info(f"Invoking MCP tool via mcpo at: POST {tool_invoke_url} with payload: {payload}")

    try:
        # Use POST method, common for REST actions / tool invocations
        response = requests.post(tool_invoke_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        logger.info(f"Received response from mcpo (Status: {response.status_code})")

        # Assuming mcpo returns the direct JSON output from the underlying MCP tool
        response_data = response.json()

        # Format the output nicely (this might need adjustment based on actual MCP server response structure)
        if isinstance(response_data, dict):
             # Simple formatting, adjust as needed based on tool output specifics
             return json.dumps(response_data, indent=2)
        elif isinstance(response_data, str):
             # If the response is already a string, return it directly
             return response_data
        else:
             # Fallback for other types
             return str(response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling mcpo wrapper for tool '{tool_name}': {e}", exc_info=True)
        error_details = str(e)
        if e.response is not None:
             try:
                  # Include response text for better debugging (e.g., detailed 404, 422 validation errors)
                  error_details = f"Status {e.response.status_code}: {e.response.text}"
             except Exception:
                  pass # Keep original error string
        # Special handling for 404 specifically on the tool path
        if e.response is not None and e.response.status_code == 404:
             return f"Error: The tool endpoint '{tool_invoke_url}' was not found on the MCP server. Check if the tool name '{tool_name}' is correct and exposed by the server."
        return f"Error communicating with the MCP server via mcpo: {error_details}"
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from mcpo for tool '{tool_name}'. Response: {response.text}")
        return f"Error: Received invalid JSON response from mcpo wrapper."
    except Exception as e:
        logger.error(f"Unexpected error invoking mcpo-wrapped MCP tool '{tool_name}': {e}", exc_info=True)
        return f"An unexpected error occurred while calling the MCP tool via mcpo: {str(e)}"


# --- GitHub Agent Tools (using MCP via mcpo) ---

def search_github_repositories_func(query: str = "") -> str:
    """
    Searches GitHub repositories using the 'search_repositories' tool via the MCP server (mcpo wrapper).
    Requires a search query.
    """
    if not query:
        return "GitHub search failed: Search query is required."

    # Tool name expected by github-mcp-server (used as path segment by mcpo)
    mcp_tool_name = "search_repositories"
    # Inputs expected by the 'search_repositories' tool (check github-mcp-server docs)
    inputs = {"query": query} # Adjusted based on search results for github-mcp-server tools

    return _invoke_mcp_tool(mcp_tool_name, inputs)

def get_github_repo_file_func(owner: str = "", repo: str = "", path: str = "") -> str:
    """
    Gets the content of a file from a GitHub repository using the 'get_file_contents' tool via the MCP server (mcpo wrapper).
    Requires repository owner, repository name, and the file path (e.g., 'README.md').
    """
    if not owner:
        return "Get file failed: Repository owner is required."
    if not repo:
        return "Get file failed: Repository name is required."
    if not path:
        return "Get file failed: File path is required."

    # Tool name expected by github-mcp-server (used as path segment by mcpo)
    mcp_tool_name = "get_file_contents"
     # Inputs expected by the 'get_file_contents' tool (check github-mcp-server docs)
    inputs = {
        "owner": owner,
        "repo": repo,
        "path": path,
    }

    return _invoke_mcp_tool(mcp_tool_name, inputs)

# Add more functions here to wrap other tools provided by the github-mcp-server as needed
# e.g., create_issue_func, list_branches_func, etc. following the pattern above.
# Ensure the 'mcp_tool_name' matches the actual tool name from github-mcp-server
# and the 'inputs' dictionary matches the arguments required by that specific tool.