#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
ENV_FILE=".env"
TRIGGER_NAME="adk-demo-main-trigger"
GITHUB_REPO_OWNER="samberthol"
GITHUB_REPO_NAME="adk-demo"
TRIGGER_BRANCH="^main$" # Regex for the main branch
BUILD_CONFIG_FILE="cloudbuild.yaml" # Assumes rename from cloudbuild-feature.yaml

# --- Helper Function to Check for Required Tools ---
check_tool() {
  command -v "$1" >/dev/null 2>&1 || { echo >&2 "Error: Required tool '$1' is not installed. Aborting."; exit 1; }
}

# --- Check Prerequisites ---
check_tool "gcloud"
check_tool "grep"
check_tool "sed"

# --- Load .env file ---
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: Environment file '$ENV_FILE' not found."
  echo "Please create it with the necessary substitution variables."
  exit 1
fi
echo "Loading environment variables from $ENV_FILE..."
set -a # Automatically export all variables
source "$ENV_FILE"
set +a
echo "Variables loaded."

# --- Check GCP Project ID is Set ---
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "Error: GCP_PROJECT_ID is not set in $ENV_FILE."
    exit 1
fi
echo "Using GCP Project: $GCP_PROJECT_ID"
gcloud config set project "$GCP_PROJECT_ID"

# --- Format substitutions from .env file ---
# Reads variables from .env, adds "_" prefix, and formats for gcloud command
echo "Formatting substitution variables..."
SUBSTITUTIONS=""
# Read lines from .env, ignore comments and empty lines
while IFS= read -r line || [[ -n "$line" ]]; do
  # Remove comments and surrounding whitespace
  line=$(echo "$line" | sed 's/#.*//' | sed 's/^[ \t]*//;s/[ \t]*$//')
  # Skip empty lines
  [ -z "$line" ] && continue
  # Split KEY=VALUE
  key=$(echo "$line" | cut -d '=' -f 1)
  value=$(echo "$line" | cut -d '=' -f 2-)
  # Add to substitutions string if key is not empty
  if [ -n "$key" ]; then
    if [ -z "$SUBSTITUTIONS" ]; then
      SUBSTITUTIONS="_${key}=${value}"
    else
      SUBSTITUTIONS="${SUBSTITUTIONS},_${key}=${value}"
    fi
  fi
done < "$ENV_FILE"

if [ -z "$SUBSTITUTIONS" ]; then
    echo "Error: Could not read any variables from $ENV_FILE to use as substitutions."
    exit 1
fi
# echo "Substitutions string: $SUBSTITUTIONS" # Uncomment for debugging

# --- Security Warning ---
echo "---------------------------------------------------------------------"
echo "SECURITY WARNING:"
echo "This script will create/update a Cloud Build trigger using substitutions"
echo "directly from the '$ENV_FILE' file. This includes potentially sensitive"
echo "values like GOOGLE_API_KEY and GITHUB_TOKEN."
echo "For production environments, it is STRONGLY recommended to store"
echo "secrets in Google Secret Manager and reference them in the Cloud Build"
echo "trigger configuration instead of passing them as direct substitutions."
echo "See: https://cloud.google.com/build/docs/securing-builds/use-secrets"
echo "---------------------------------------------------------------------"
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo # Move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborting."
    exit 1
fi

# --- Connect to GitHub Repository (User Interaction Needed First Time) ---
echo "Checking GitHub connection..."
# This command might fail if connection already exists, which is okay.
# It primarily helps initiate the connection process if needed.
gcloud alpha builds connections create github "$GITHUB_REPO_OWNER-$GITHUB_REPO_NAME-connection" --region="$REGION" --authorizer-token-secret-version=latest --app-installation-id=0 || echo "GitHub connection likely already exists or setup needs manual completion in browser."

echo "---------------------------------------------------------------------"
echo "IMPORTANT: Ensure the Cloud Build GitHub App is installed on the"
echo "'$GITHUB_REPO_OWNER/$GITHUB_REPO_NAME' repository or the '$GITHUB_REPO_OWNER' organization."
echo "You might have been prompted to install/authorize it in your browser"
echo "during the 'gcloud alpha builds connections create' step if it wasn't connected."
echo "Verify installation here: https://github.com/settings/installations"
echo "---------------------------------------------------------------------"
read -p "Press Enter to continue once the GitHub App is installed/verified..."

# --- Create/Update Cloud Build Trigger ---
echo "Attempting to create/update Cloud Build trigger '$TRIGGER_NAME'..."

# --- This is the CORRECTED block for your script ---
if gcloud beta builds triggers describe "$TRIGGER_NAME" --region="$REGION" > /dev/null 2>&1; then
    echo "Trigger '$TRIGGER_NAME' already exists. Updating it..."
    gcloud beta builds triggers update github "$TRIGGER_NAME" \
      --region="$REGION" \
      --repo-owner="$GITHUB_REPO_OWNER" \
      --repo-name="$GITHUB_REPO_NAME" \
      --branch-pattern="$TRIGGER_BRANCH" \
      --build-config="$BUILD_CONFIG_FILE" \
      --update-substitutions="$SUBSTITUTIONS" \
      --description="Trigger for ADK Demo Main Branch" \
      --project="$GCP_PROJECT_ID"
else
    echo "Trigger '$TRIGGER_NAME' does not exist. Creating it..."
    gcloud beta builds triggers create github \
      --name="$TRIGGER_NAME" \
      --region="$REGION" \
      --repo-owner="$GITHUB_REPO_OWNER" \
      --repo-name="$GITHUB_REPO_NAME" \
      --branch-pattern="$TRIGGER_BRANCH" \
      --build-config="$BUILD_CONFIG_FILE" \
      --substitutions="$SUBSTITUTIONS" \
      --description="Trigger for ADK Demo Main Branch" \
      --project="$GCP_PROJECT_ID"
fi

echo "Cloud Build trigger '$TRIGGER_NAME' setup complete."
echo "It will now trigger a build when changes are pushed to the '$TRIGGER_BRANCH' branch of '$GITHUB_REPO_OWNER/$GITHUB_REPO_NAME'."