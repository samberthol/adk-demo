# agents/datascience/tools.py
import google.cloud.bigquery
import logging
import pandas as pd
import os
from typing import Optional
from google.api_core.exceptions import Conflict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_bigquery_query_func(
    project_id: Optional[str] = None, # <-- Changed default to None
    query: str = "" # Keep query required
    ) -> str:
    """Executes a SQL query against BigQuery and returns the results as a string."""

    # Fetch project_id from environment or use fallback
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', None) # Fallback to None

    # Ensure project_id is provided (either via arg or env var)
    if not project_id:
         return "BigQuery query failed: Project ID is required (must be provided as argument or set in GCP_PROJECT_ID env var)."
    if not query:
         return "BigQuery query failed: Query string is required."

    try:
        bq_client = google.cloud.bigquery.Client(project=project_id)
        logger.info(f"Executing BigQuery query in project {project_id}: {query}")

        query_job = bq_client.query(query)
        results = query_job.result()

        df = results.to_dataframe()
        if df.empty:
            return "Query executed successfully, but returned no results."
        else:
            # Consider limiting the size of the returned string for very large results
            # return df.to_string(index=False, max_rows=100)
            return df.to_string(index=False)

    except Exception as e:
        logger.error(f"Failed to execute BigQuery query in project {project_id}: {e}", exc_info=True)
        error_message = getattr(e, 'message', str(e))
        # Extract more specific error details if available
        if hasattr(e, 'errors') and e.errors:
             try:
                  error_details = e.errors[0].get('message', error_message)
             except (IndexError, TypeError, AttributeError, KeyError):
                  error_details = str(e.errors) # Fallback to string representation of errors list
             return f"BigQuery query failed in project {project_id}: {error_details}"
        # Return generic error if specific details aren't available
        return f"BigQuery query failed in project {project_id}: {error_message}"

# --- BigQuery Dataset Tool ---
def create_bq_dataset_func(
    project_id: Optional[str] = None, # <-- Changed default to None
    dataset_id: str = "", # Keep dataset_id required
    location: Optional[str] = None # <-- Changed default to None
    ) -> str:
    """Creates a BigQuery dataset."""

    # Fetch values from environment or use fallbacks
    project_id = project_id or os.environ.get('GCP_PROJECT_ID', None) # Fallback to None
    location = location or os.environ.get('BQ_DEFAULT_LOCATION', 'US') # Use specific env var or fallback

    # Ensure project_id and dataset_id are provided
    if not project_id:
         return "Failed to create BigQuery dataset: Project ID is required (must be provided as argument or set in GCP_PROJECT_ID env var)."
    if not dataset_id:
         return "Failed to create BigQuery dataset: Dataset ID is required."


    try:
        bq_client = google.cloud.bigquery.Client(project=project_id)
        full_dataset_id = f"{project_id}.{dataset_id}" # Construct full ID here

        dataset = google.cloud.bigquery.Dataset(full_dataset_id)
        dataset.location = location # Use the resolved location

        logger.info(f"Creating BigQuery dataset {full_dataset_id} in location {location}...")
        created_dataset = bq_client.create_dataset(dataset, timeout=30)
        logger.info(f"Dataset {created_dataset.dataset_id} created successfully.")
        return f"BigQuery dataset {created_dataset.dataset_id} created successfully in project '{project_id}', location '{location}'."

    except Conflict:
         logger.warning(f"Dataset {full_dataset_id} already exists.")
         # Use full_dataset_id for clarity
         return f"BigQuery dataset '{full_dataset_id}' already exists."
    except Exception as e:
        logger.error(f"Failed to create BigQuery dataset {dataset_id} in project {project_id}: {e}", exc_info=True)
        # Use full_dataset_id for clarity
        return f"Failed to create BigQuery dataset '{full_dataset_id}': {str(e)}"