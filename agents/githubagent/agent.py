# adk-demo/agents/githubagent/agent.py
import os
from google.adk.agents import LlmAgent
from .tools import search_github_repositories_func, get_github_repo_file_func

# Read model name from environment variable, with a fallback
agent_model = os.environ.get('AGENT_MODEL_NAME', 'gemini-2.0-flash')

githubagent = LlmAgent(
    name="githubagent", # Lowercase as requested
    model=agent_model,
    description="An agent specialized in interacting with GitHub repositories using tools that communicate with a GitHub MCP server.",
    instruction=(
        "You are a specialized agent that uses tools to interact with GitHub via a Model Context Protocol (MCP) server.\n\n"
        "Available Tools:\n"
        "1.  **`search_github_repositories_func`**: Searches GitHub for repositories matching a query. Requires `query` string.\n"
        "2.  **`get_github_repo_file_func`**: Gets the content of a specific file (like a README) from a GitHub repository. Requires `owner`, `repo`, and `path`.\n\n"
        "Analyze the user's request:\n"
        "- If the user wants to **search for repositories**, call `search_github_repositories_func`. Make sure you have the search query.\n"
        "- If the user wants to **get a file from a repository**, call `get_github_repo_file_func`. Make sure you have the repository owner, name, and the file path.\n"
        "- Provide the results or status from the tool calls clearly back to the user.\n"
        "- You interact with a backend MCP server; do not mention MCP to the user, just use the tools."
    ),
    tools=[
        search_github_repositories_func,
        get_github_repo_file_func,
    ],
)