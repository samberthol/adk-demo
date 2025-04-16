# GCP Multi-Agent Demo using Google ADK and MCP

This project demonstrates a multi-agent system built with the Google Agent Development Kit (ADK), showcasing interactions with Google Cloud services (Compute Engine, BigQuery) and GitHub. It features automated CI/CD deployment to Google Cloud Run using Cloud Build triggers.

## Overview

This project utilizes a `MetaAgent` to intelligently route user requests to specialized agents responsible for specific tasks:
* **Resource Management:** Interacting with Google Compute Engine VMs.
* **Data Science:** Querying Google BigQuery datasets.
* **GitHub:** Interacting with GitHub repositories via a MCP proxy.

The system includes a web-based UI built with Streamlit for interaction. Deployment is fully automated via Cloud Build upon pushes to the main branch of the configured GitHub repository.

## Core Components

This demo integrates the following solutions :

* **[Google Agent Development Kit (ADK)](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-development-kit/quickstart):** The core Python framework for building the multi-agent system.
* **Specialized Agents:**
    * `MetaAgent`: Orchestrates requests to sub-agents.
    * `ResourceAgent`: Manages Google Compute Engine resources. Uses the [Compute Engine API](https://cloud.google.com/compute/docs).
    * `DataScienceAgent`: Manages Google BigQuery resources. Uses the [BigQuery API](https://cloud.google.com/bigquery/docs).
    * `githubagent`: Interacts with GitHub via the MCP server.
* **User Interface:**
    * **[Streamlit](https://docs.streamlit.io/):** Provides a rapid development web UI for chat interaction.
* **[GitHub MCP Server](https://github.com/github/github-mcp-server):** A Go application (running in a separate container) acting as a proxy to provide GitHub tools via the Model Context Protocol (MCP), wrapped by `mcpo` for HTTP access. See also [MCP Server Examples](https://github.com/modelcontextprotocol/servers).
* **[Google Cloud Build](https://cloud.google.com/build/docs):** Automates the build, test, and deployment pipeline defined in `cloudbuild.yaml`.
* **[Google Cloud Run](https://cloud.google.com/run/docs):** Hosts the containerized applications (Streamlit UI, MCP Server) as scalable, managed services.
* **[Google Artifact Registry](https://cloud.google.com/artifact-registry/docs):** Stores the built container images.

**Project Diagram**

<p align="center">
<img src="./assets/td-flow-chart.png" alt="Diagram" width="800"/>
</p>

## Prerequisites

Before you begin, ensure you have the following prerequisites:

* **Local Tools:**
    * Bash (or a compatible shell like zsh).
    * [Git](https://git-scm.com/) installed.
    * [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and authenticated (`gcloud auth login`, `gcloud auth application-default login`).
* **Google Cloud:**
    * A Google Cloud Project with billing enabled.
    * The following APIs enabled in your project:
        * Compute Engine API
        * BigQuery API
        * Cloud Build API
        * Cloud Run API
        * Artifact Registry API
        * IAM Service Account Credentials API (often needed by Cloud Build SA)
        * Secret Manager API (Recommended for storing secrets).
    * An [Artifact Registry Docker repository](https://cloud.google.com/artifact-registry/docs/repositories/create-repos#docker) created in the desired region.
    * A user or service account with sufficient IAM permissions. The Cloud Build service account (`PROJECT_NUMBER@cloudbuild.gserviceaccount.com`) typically needs roles like:
        * `Cloud Build Service Account`
        * `Cloud Run Admin`
        * `Artifact Registry Writer`
        * `Service Account User` (to act as the Cloud Run runtime service account)
        * `IAM Service Account Token Creator`
        * Permissions to access any other services the agents use (e.g., `roles/bigquery.user`, `roles/compute.instanceAdmin.v1`).
* **GitHub:**
    * A GitHub account.
    * A [GitHub Personal Access Token (Classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) with `repo` scope (required by the MCP server and potentially the trigger setup script).
    * The [Cloud Build GitHub App](https://github.com/apps/google-cloud-build) installed and configured for the repository you intend to use (your fork).

## Deployment

To deploy this project using your own GCP project and GitHub fork:

1.  **Fork the Repository:** Create a fork of the original `samberthol/adk-demo` repository under your own GitHub account.

2.  **Clone Your Fork:** Clone the repository you just forked to your local machine:
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/adk-demo.git](https://github.com/YOUR_GITHUB_USERNAME/adk-demo.git)
    cd adk-demo
    ```
    *(Replace `YOUR_GITHUB_USERNAME`)*

3.  **Prepare Environment Variables (`.env`):**
    * Copy `.env.example` to `.env`.
    * **Crucially, edit `.env` and fill in *your specific values*** for:
        * `GCP_PROJECT_ID`: Your Google Cloud project ID.
        * `REGION`: The GCP region (e.g., `us-central1`) where you created the Artifact Registry repo and want to deploy Cloud Run services.
        * `REPO_ID`: The name of your Artifact Registry Docker repository.
        * `GITHUB_REPO_OWNER`: Your GitHub username (the owner of the fork).
        * `GITHUB_REPO_NAME`: The name of your forked repository (usually `adk-demo`).
        * `GOOGLE_API_KEY`: Your Google API Key (for Generative AI access).
        * `GITHUB_TOKEN`: Your GitHub Personal Access Token (Classic) with `repo` scope.
        * `PROJECT_HASH`: The Project Number associated with Cloud Run URLs in your project. Find this by deploying *any* service to Cloud Run in your project/region via the console; the hash is part of the generated URL (e.g., `service-name-abcdefgh-uc.a.run.app` -> hash is `abcdefgh`).
        * Other variables as needed (review `.env.example` and `cloudbuild.yaml` substitutions).
    * ***Security Note:*** For production environments, avoid storing `GOOGLE_API_KEY` and `GITHUB_TOKEN` directly in `.env`. Use [Google Secret Manager](https://cloud.google.com/secret-manager) and reference the secrets in `cloudbuild.yaml`. You would need to adjust `setup_trigger.sh` to *not* pass these as substitutions if using Secret Manager.

4.  **Configure and Run the Trigger Setup Script:**
    * Review `setup_trigger.sh`. You might need to update `GITHUB_REPO_OWNER` and `GITHUB_REPO_NAME` variables within the script itself if they differ from your `.env` file (though ideally they match).
    * Make the script executable: `chmod +x setup_trigger.sh`
    * Run the script: `./setup_trigger.sh`.
    * This script will:
        * Set your `gcloud` context to the project specified in `.env`.
        * Attempt to connect GCP Cloud Build to your GitHub account/repository. **This may require interaction in your web browser** the first time to authorize the Google Cloud Build GitHub App for your repository fork. Follow the prompts carefully.
        * Create or update the Cloud Build trigger (`adk-demo-main-trigger` by default) linked to the `main` branch of *your specified repository*.
        * Configure the trigger to use `cloudbuild.yaml` and inject substitutions from your `.env` file (respecting the security warning about secrets).

5.  **Trigger the First Build & Deployment:**
    * Commit and push any changes (including your updated `.env` if you track it - *not recommended for sensitive data*) to the `main` branch of *your forked repository*.
    * This push will activate the Cloud Build trigger you just set up.
    * Monitor the build progress in the Google Cloud Console: Cloud Build -> History.

6.  **First Build Considerations:**
    * The **very first Cloud Build *run*** triggered by your push might require attention:
        * **GitHub Connection:** Ensure you completed any browser-based authorization steps prompted during the `setup_trigger.sh` execution. If the build fails with connection errors, double-check the Cloud Build GitHub App installation and permissions on your fork.
        * **IAM Permissions:** The first run might fail if the Cloud Build service account (`PROJECT_NUMBER@cloudbuild.gserviceaccount.com`) doesn't yet have all necessary permissions (e.g., to push to Artifact Registry, deploy to Cloud Run, act as the Cloud Run service account). Check the build logs for permission errors and grant the required roles in the IAM section of the GCP console. Sometimes permissions take a few moments to propagate.
    * If the first build fails, troubleshoot using the build logs, fix the underlying issue (permissions, configuration), and trigger a new build by pushing another small change to your `main` branch.

7.  **Access Deployed Services:**
    * Once the Cloud Build pipeline succeeds, it will deploy two Cloud Run services.
    * Find the public URL for the Streamlit (`SERVICE_NAME_STREAMLIT`) service in the Cloud Run section of the GCP console. The MCP service (`SERVICE_NAME_MCP`) runs internally and is accessed by the Streamlit service via its service URL.

## Configuration Notes

* **Secrets Management:** Using Google Secret Manager is the recommended practice for handling sensitive data like API keys and tokens in `cloudbuild.yaml`, rather than passing them as direct substitutions from `.env`.
* **Session Persistence:** The UIs currently use ADK's `InMemorySessionService`. This means conversation history is lost if the Cloud Run instances restart.

## Contributing

All contributions are welcome, feedback is a gift !