# agents/datascience/agent.py
from google.adk.agents import LlmAgent
# Import the functions directly, including the new create function
from .tools import run_bigquery_query_func, create_bq_dataset_func # <-- Added create_bq_dataset_func

data_science_agent = LlmAgent(
    name="DataScienceAgent",
    model="gemini-2.0-flash", # Ensure this model supports function calling well
    description="An agent specialized in querying data from Google BigQuery using SQL and creating BigQuery datasets.", # Added creating datasets
    instruction=(
        "You are a data analyst agent responsible for interacting with Google BigQuery.\n"
        "You have two tools available:\n"
        "1.  **`run_bigquery_query_func`**: Use this to execute SQL queries. Requires `project_id` and the SQL `query` string.\n"
        "2.  **`create_bq_dataset_func`**: Use this to create a new BigQuery dataset. Requires `project_id` and `dataset_id`. It accepts an optional `location` (defaults to 'US').\n\n"
        "Analyze the user's request:\n"
        "- If the user wants to **query data**, call `run_bigquery_query_func`. Make sure you have the project ID and the query.\n"
        "- If the user wants to **create a dataset**, call `create_bq_dataset_func`. Make sure you have the project ID and the desired dataset name.\n"
        "- Provide the results or status from the tool calls clearly back to the user."
    ),
    # Add the create function to the tools list
    tools=[
        run_bigquery_query_func,
        create_bq_dataset_func # <-- Added create tool
        ],
)