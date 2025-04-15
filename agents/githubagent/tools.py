# adk-demo/agents/githubagent/tools.py
import os
import requests
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Get MCP Server URL and GitHub Token from environment variables
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") # May be needed by MCP server or direct calls

# Define the base URL for invoking tools on the MCP server
# Adjust '/invoke' if the github-mcp-server uses a different endpoint path
MCP_INVOKE_URL = f"{MCP_SERVER_URL}/invoke" if MCP_SERVER_URL else None

# --- Helper Function to Call MCP Server ---
def _invoke_mcp_tool(tool_name: str, inputs: dict) -> str:
    """Helper function to invoke a tool on the configured MCP server via HTTP POST."""
    if not MCP_INVOKE_URL:
        return "Error: MCP_SERVER_URL environment variable is not set. Cannot contact MCP server."

    headers = {
        "Content-Type": "application/json",
        # The github-mcp-server might require the token here or handle it internally
        # Check its documentation. For simplicity, we assume internal handling for now.
        # If needed: "Authorization": f"Bearer {GITHUB_TOKEN}"
    }
    payload = {
        "tool": tool_name,
        "inputs": inputs
    }

    logger.info(f"Invoking MCP tool '{tool_name}' at {MCP_INVOKE_URL} with inputs: {inputs}")

    try:
        response = requests.post(MCP_INVOKE_URL, headers=headers, json=payload, timeout=60) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        logger.info(f"Received response from MCP server (Status: {response.status_code})")
        # Assuming the MCP server returns JSON with an 'outputs' field
        response_data = response.json()
        outputs = response_data.get("outputs", {})

        # Format the output nicely (this might need adjustment based on actual MCP server response structure)
        if isinstance(outputs, dict):
             # Simple formatting, adjust as needed based on tool output specifics
             return json.dumps(outputs, indent=2)
        else:
             return str(outputs) # Fallback for non-dict outputs

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling MCP server for tool '{tool_name}': {e}", exc_info=True)
        error_details = str(e)
        if e.response is not None:
             try:
                  error_details = f"Status {e.response.status_code}: {e.response.text}"
             except Exception:
                  pass # Keep original error string
        return f"Error communicating with MCP server: {error_details}"
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON response from MCP server for tool '{tool_name}'. Response: {response.text}")
        return f"Error: Received invalid response from MCP server."
    except Exception as e:
        logger.error(f"Unexpected error invoking MCP tool '{tool_name}': {e}", exc_info=True)
        return f"An unexpected error occurred while calling the MCP tool: {str(e)}"


# --- GitHub Agent Tools (using MCP) ---

def search_github_repositories_func(query: str = "") -> str:
    """
    Searches GitHub repositories using the 'search_repositories' tool via the MCP server.
    Requires a search query.
    """
    if not query:
        return "GitHub search failed: Search query is required."

    # Tool name expected by github-mcp-server (verify from its docs)
    mcp_tool_name = "search_repositories"
    inputs = {"query": query}

    return _invoke_mcp_tool(mcp_tool_name, inputs)

def get_github_repo_file_func(owner: str = "", repo: str = "", path: str = "") -> str:
    """
    Gets the content of a file from a GitHub repository using the 'get_file_contents' tool via the MCP server.
    Requires repository owner, repository name, and the file path (e.g., 'README.md').
    """
    if not owner:
        return "Get file failed: Repository owner is required."
    if not repo:
        return "Get file failed: Repository name is required."
    if not path:
        return "Get file failed: File path is required."

    # Tool name expected by github-mcp-server (verify from its docs)
    mcp_tool_name = "get_file_contents" # Or maybe "get_repo_content" - check MCP server tool list
    inputs = {
        "owner": owner,
        "repo": repo,
        "path": path,
        # 'branch': 'main' # Optionally add branch if needed
    }

    return _invoke_mcp_tool(mcp_tool_name, inputs)

# Add more functions here to wrap other tools provided by the github-mcp-server as needed
# e.g., create_issue_func, list_branches_func, etc. following the pattern above.