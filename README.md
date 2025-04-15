# GCP Multi-Agent Demo (using Google ADK)

This project demonstrates a multi-agent system built with the Google Agent Development Kit (ADK), deployed automatically via Cloud Build triggers. It uses a `MetaAgent` to coordinate tasks between specialized agents that interact with Google Cloud services and GitHub.

## Overview

The system features:

* **`MetaAgent`**: Routes user requests to the appropriate specialized agent.
* **`ResourceAgent`**: Manages [Google Compute Engine](https://cloud.google.com/compute/docs) VM instances (create, delete, list, start, stop, get details).
* **`DataScienceAgent`**: Interacts with [Google BigQuery](https://cloud.google.com/bigquery/docs) (runs queries, creates datasets).
* **`githubagent`**: Interacts with GitHub via a separate MCP (Model Context Protocol) server proxy.
* **Streamlit UI**: Provides a web interface for interacting with the agents.
* **FastAPI UI**: Provides an alternative API/WebSocket interface (e.g., for voice interaction).
* **MCP Server**: A separate service acting as a proxy for GitHub tools.
* **Automated Deployment**: Uses [Google Cloud Build](https://cloud.google.com/build/docs) triggers connected to GitHub for automated builds and deployments to [Google Cloud Run](https://cloud.google.com/run). Container images are stored in [Google Artifact Registry](https://cloud.google.com/artifact-registry/docs).

**Project Diagram** : Workflow of the different component (Update if necessary)

<p align="center">
<img src="./assets/td-flow-chart.png" alt="Diagram" width="600"/>
</p>

## Project Structure (Adjust if needed after cleanup)

    adk-demo/
    ├── agents/                 # Agent definitions and tools
    │   ├── datascience/
    │   ├── githubagent/
    │   ├── meta/
    │   └── resource/
    ├── assets/
    │   └── td-flow-chart.png   # Diagram
    ├── ui/                     # UI applications
    │   ├── app.py              # Streamlit UI
    │   └── api.py              # FastAPI UI
    ├── main.py                 # Main entry point / Root agent definition
    ├── requirements.txt        # Python dependencies
    ├── Dockerfile              # Dockerfile for Python App (Streamlit/FastAPI)
    ├── Dockerfile.mcp          # Dockerfile for MCP Go server (Ensure this exists)
    ├── cloudbuild.yaml         # Cloud Build configuration (Multi-service)
    ├── setup_trigger.sh        # Script to create/update the Cloud Build GitHub trigger
    ├── .env.example            # Example environment variables file
    ├── .gitignore              # Files ignored by Git
    ├── LICENSE                 # Project license
    └── README.md               # This file

## Prerequisites

* Bash
* [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and authenticated.
* [Git](https://git-scm.com/) installed.
* A Google Cloud Project with the following APIs enabled:
    * Compute Engine API
    * BigQuery API
    * Cloud Build API
    * Cloud Run API
    * Artifact Registry API
    * Secret Manager API (Recommended for secrets)
* A user or service account with necessary permissions (e.g., Cloud Build Editor, Cloud Run Admin, Artifact Registry Writer, Service Account User, Secret Manager Secret Accessor if using secrets).
* An Artifact Registry Docker repository in your project/region.
* A [Google API Key](https://aistudio.google.com/apikey).
* A [GitHub Personal Access Token (Classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) with `repo` scope (or finer-grained permissions if possible).
* The [Cloud Build GitHub App](https://github.com/apps/google-cloud-build) installed on your GitHub repository (`samberthol/adk-demo`) or organization (`samberthol`).

## Setup and Automated Deployment

This project uses Cloud Build triggers for automated deployment upon pushes to the `main` branch.

1.  **Clone the repository:**

        git clone https://github.com/samberthol/adk-demo.git
        cd adk-demo

2.  **Prepare Environment Variables:**
    * Copy `.env.example` to `.env`.
    * Fill in all the required values in the `.env` file (e.g., `GCP_PROJECT_ID`, `REGION`, `GOOGLE_API_KEY`, `GITHUB_TOKEN`, `PROJECT_HASH`, etc.).
    * ***Security Note:*** For production, manage sensitive values like `GOOGLE_API_KEY` and `GITHUB_TOKEN` securely using [Secret Manager](https://cloud.google.com/secret-manager) instead of placing them directly in `.env`. Modify `setup_trigger.sh` and `cloudbuild.yaml` accordingly if using Secret Manager.

3.  **Ensure GitHub App Installation:**
    * Verify the [Cloud Build GitHub App](https://github.com/apps/google-cloud-build) is installed and has access to your `samberthol/adk-demo` repository. You can check under your GitHub profile -> Settings -> Applications -> Installed GitHub Apps.

4.  **Set up the Cloud Build Trigger:**
    * Make the setup script executable: `chmod +x setup_trigger.sh`
    * Run the script: `./setup_trigger.sh`
    * This script will:
        * Connect your GCP project to your GitHub repository (may require browser interaction on first run).
        * Create or update a Cloud Build trigger named `adk-demo-main-trigger`.
        * Configure the trigger to use `cloudbuild.yaml` and inject substitutions from your `.env` file. Follow the script prompts carefully.

5.  **Trigger Deployment:**
    * Ensure your `feature/live-voice-connect` branch is merged into `main`.
    * Push changes to the `main` branch of your GitHub repository (`samberthol/adk-demo`).
    * This push will automatically trigger the Cloud Build process defined in `cloudbuild.yaml`.
    * Monitor the build progress in the Google Cloud Console under Cloud Build -> History.
    * Once the build completes successfully, the three Cloud Run services (Streamlit, FastAPI, MCP) will be deployed or updated. Find their URLs in the Cloud Run section of the console.

## Configuration Notes

* **Secrets Management:** As mentioned, using Google Secret Manager for API keys and tokens is highly recommended over `.env` substitutions for better security.
* **Project Hash:** The `_PROJECT_HASH` substitution in `cloudbuild.yaml` (and corresponding `PROJECT_HASH` in `.env`) is required to construct the correct Cloud Run service URLs. You can find this hash as part of the URL of any existing Cloud Run service in your project.
* **Session Persistence:** The Streamlit and FastAPI UIs use `InMemorySessionService` by default. Chat/interaction history will be lost if the Cloud Run instances restart. For persistent sessions, explore other ADK session services.

## Contributing

Contributions are welcome! Please follow standard coding practices.