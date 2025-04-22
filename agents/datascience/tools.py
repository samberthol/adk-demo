# agents/datascience/tools.py
import google.cloud.bigquery
import logging
import os
from typing import Optional
from google.api_core.exceptions import Conflict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_bigquery_query_func(
    project_id: Optional[str] = None,
    query: str = ""
    ) -> str:

    project_id = project_id or os.environ.get('GCP_PROJECT_ID', None)

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
            return df.to_string(index=False)

    except Exception as e:
        logger.error(f"Failed to execute BigQuery query in project {project_id}: {e}", exc_info=True)
        error_message = getattr(e, 'message', str(e))
        if hasattr(e, 'errors') and e.errors:
             try:
                  error_details = e.errors[0].get('message', error_message)
             except (IndexError, TypeError, AttributeError, KeyError):
                  error_details = str(e.errors)
             return f"BigQuery query failed in project {project_id}: {error_details}"
        return f"BigQuery query failed in project {project_id}: {error_message}"

def create_bq_dataset_func(
    project_id: Optional[str] = None,
    dataset_id: str = "",
    location: Optional[str] = None
    ) -> str:

    project_id = project_id or os.environ.get('GCP_PROJECT_ID', None)
    location = location or os.environ.get('BQ_DEFAULT_LOCATION', 'US')

    if not project_id:
         return "Failed to create BigQuery dataset: Project ID is required (must be provided as argument or set in GCP_PROJECT_ID env var)."
    if not dataset_id:
         return "Failed to create BigQuery dataset: Dataset ID is required."

    try:
        bq_client = google.cloud.bigquery.Client(project=project_id)
        full_dataset_id = f"{project_id}.{dataset_id}"

        dataset = google.cloud.bigquery.Dataset(full_dataset_id)
        dataset.location = location

        logger.info(f"Creating BigQuery dataset {full_dataset_id} in location {location}...")
        created_dataset = bq_client.create_dataset(dataset, timeout=30)
        logger.info(f"Dataset {created_dataset.dataset_id} created successfully.")
        return f"BigQuery dataset {created_dataset.dataset_id} created successfully in project '{project_id}', location '{location}'."

    except Conflict:
         logger.warning(f"Dataset {full_dataset_id} already exists.")
         return f"BigQuery dataset '{full_dataset_id}' already exists."
    except Exception as e:
        logger.error(f"Failed to create BigQuery dataset {dataset_id} in project {project_id}: {e}", exc_info=True)
        return f"Failed to create BigQuery dataset '{full_dataset_id}': {str(e)}"