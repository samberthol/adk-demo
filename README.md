# GCP Multi-Agent Demo (using Google ADK)

This project demonstrates a multi-agent system built with the Google Agent Development Kit (ADK). It uses a `MetaAgent` to coordinate tasks between specialized agents that interact with Google Cloud services.

## Overview

The system features:

* **`ResourceAgent`**: Manages [Google Compute Engine](https://cloud.google.com/compute/docs) VM instances (create, delete, list, start, stop, get details).
* **`DataScienceAgent`**: Interacts with [Google BigQuery](https://cloud.google.com/bigquery/docs) (runs queries, creates datasets).
* **`MetaAgent`**: Routes user requests to the appropriate specialized agent.
* **Streamlit UI**: Provides a web interface for interacting with the agents.
* **Deployment Scripts**: Automates building and deployment using [Google Cloud Build](https://cloud.google.com/build/docs) and [Google Cloud Run](https://cloud.google.com/run). Container images are stored in [Google Artifact Registry](https://cloud.google.com/artifact-registry/docs).

## Project Structure

    google-agentic/
    ├── agents/               # Agent definitions and tools
    │   ├── datascience/
    │   │   ├── agent.py      # DataScience Agent
    │   │   └── tools.py      # BigQuery tools
    │   ├── meta/
    │   │   └── agent.py      # Meta Agent (orchestrator)
    │   └── resource/
    │       ├── agent.py      # Resource Agent
    │       └── tools.py      # Compute Engine tools
    ├── ui/                   # Streamlit UI application
    │   └── app.py
    ├── main.py               # Main entry point / Root agent definition
    ├── requirements.txt      # Python dependencies
    ├── Dockerfile            # Dockerfile for containerization
    ├── cloudbuild.yaml       # Cloud Build configuration
    ├── build_then_run.sh     # Build and deploy script
    ├── .gitignore            # Files ignored by Git
    ├── LICENSE               # Project license
    └── README.md             # This file

## Prerequisites

* Python 3.9+
* [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and authenticated.
* A Google Cloud Project with the following APIs enabled:
    * Compute Engine API
    * BigQuery API
    * Cloud Build API
    * Cloud Run API
    * Artifact Registry API
* A [Google API Key](https://cloud.google.com/docs/authentication/api-keys) (for ADK interaction with services like Gemini models).

## Setup

1.  **Clone the repository:**

        git clone <your-repository-url>
        cd google-agentic

2.  **Create and activate a virtual environment:**

        python3 -m venv venv
        source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3.  **Set Environment Variables:**
    * Create a `.env` file in the project root or export these variables:

            # Required for ADK/Gemini
            export GOOGLE_API_KEY='YOUR_ACTUAL_API_KEY_HERE'

            # Required for deployment script and GCP tools (if not overridden in code)
            export GCP_PROJECT_ID='your-gcp-project-id'

            # Optional: Set defaults for VM/BQ tools if needed
            # export VM_DEFAULT_ZONE='us-central1-c'
            # export BQ_DEFAULT_LOCATION='US'

    * ***Security Note:*** For production, manage `GOOGLE_API_KEY` securely using tools like [Secret Manager](https://cloud.google.com/secret-manager).

## Deployment to Cloud Run

The `build_then_run.sh` script automates the build and deployment process:

1.  **Ensure Prerequisites:** Make sure `gcloud` is authenticated and configured for the correct project (`gcloud config set project YOUR_GCP_PROJECT_ID`). Ensure `GOOGLE_API_KEY` and `GCP_PROJECT_ID` are set in your environment or `.env` file.
2.  **Make script executable:**

        chmod +x build_then_run.sh

3.  **Run the script:**

        ./build_then_run.sh

This script uses Cloud Build to create a Docker image, pushes it to Artifact Registry, and deploys it as a service on Cloud Run.

## Configuration Notes

* **Defaults:** Default values for GCP project ID, zone, machine types, etc., are often set within the `agents/*/tools.py` files, potentially falling back to environment variables. Consider centralizing configuration for better flexibility.
* **Session Persistence:** The Streamlit UI uses `InMemorySessionService` by default. Chat history will be lost if the Cloud Run instance restarts. For persistent sessions, explore other ADK session services documented in the [Agent Development Kit](https://google.github.io/adk-docs/).

## Contributing

Contributions are welcome! Please follow standard coding practices. (Consider adding a `CONTRIBUTING.md` for detailed guidelines).