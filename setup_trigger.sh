#!/bin/bash
# setup_trigger.sh

set -e

ENV_FILE=".env"

check_tool() {
  command -v "$1" >/dev/null 2>&1 || { echo >&2 "Error: Required tool '$1' is not installed. Aborting."; exit 1; }
}

check_tool "gcloud"
check_tool "grep"
check_tool "sed"

if [ ! -f "$ENV_FILE" ]; then
  echo "Error: Environment file '$ENV_FILE' not found."
  echo "Please create it with the necessary substitution and configuration variables."
  exit 1
fi
echo "Loading environment variables from $ENV_FILE..."
set -a
source "$ENV_FILE"
set +a
echo "Variables loaded."

REQUIRED_VARS=(
    "GCP_PROJECT_ID"
    "REGION"
    "TRIGGER_NAME"
    "GITHUB_REPO_OWNER"
    "GITHUB_REPO_NAME"
    "TRIGGER_BRANCH"
    "BUILD_CONFIG_FILE"
    "IGNORED_FILES"
)
MISSING_VARS=()
for VAR_NAME in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!VAR_NAME}" ]; then
        MISSING_VARS+=("$VAR_NAME")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo "Error: The following required variables are not set in '$ENV_FILE':" >&2
    for MISSING_VAR in "${MISSING_VARS[@]}"; do
        echo "  - $MISSING_VAR" >&2
    done
    exit 1
fi

echo "Using GCP Project: $GCP_PROJECT_ID"
gcloud config set project "$GCP_PROJECT_ID"

echo "Formatting substitution variables..."
SUBSTITUTIONS=""
while IFS= read -r line || [[ -n "$line" ]]; do
  line=$(echo "$line" | sed 's/#.*//' | sed 's/^[ \t]*//;s/[ \t]*$//')
  [ -z "$line" ] && continue
  key=$(echo "$line" | cut -d '=' -f 1)
  value=$(echo "$line" | cut -d '=' -f 2-)

  if [[ "$key" == _* || \
        "$key" == "GCP_PROJECT_ID" || \
        "$key" == "PROJECT_NUMBER" || \
        "$key" == "REGION" || \
        "$key" == "REPO_ID" || \
        "$key" == "PYTHON_APP_IMAGE_NAME" || \
        "$key" == "MCP_IMAGE_NAME" || \
        "$key" == "LANGGRAPH_IMAGE_NAME" || \
        "$key" == "SERVICE_NAME_STREAMLIT" || \
        "$key" == "SERVICE_NAME_MCP" || \
        "$key" == "SERVICE_NAME_LANGGRAPH" || \
        "$key" == "GOOGLE_API_KEY" || \
        "$key" == "GITHUB_TOKEN" || \
        "$key" == "AGENT_MODEL_NAME" || \
        "$key" == "BQ_DEFAULT_LOCATION" || \
        "$key" == "VM_DEFAULT_ZONE" || \
        "$key" == "VM_DEFAULT_INSTANCE_NAME" || \
        "$key" == "VM_DEFAULT_MACHINE_TYPE" || \
        "$key" == "VM_DEFAULT_SOURCE_IMAGE" || \
        "$key" == "VM_DEFAULT_DISK_SIZE_GB" || \
        "$key" == "VM_DEFAULT_DISK_TYPE" || \
        "$key" == "VM_DEFAULT_SUBNETWORK" || \
        "$key" == "VM_DEFAULT_SERVICE_ACCOUNT" || \
        "$key" == "MCP_SERVER_PORT" || \
        "$key" == "LANGGRAPH_PORT" || \
        "$key" == "MISTRAL_MODEL_ID" ]]; then
      formatted_key="_${key}"
      if [ -n "$key" ]; then
        if [ -z "$SUBSTITUTIONS" ]; then
          SUBSTITUTIONS="${formatted_key}=${value}"
        else
          SUBSTITUTIONS="${SUBSTITUTIONS},${formatted_key}=${value}"
        fi
      fi
  fi
done < "$ENV_FILE"

if [ -z "$SUBSTITUTIONS" ]; then
    echo "Warning: Could not read any variables designated for substitutions from $ENV_FILE."
fi

# --- Security Warning ---
echo "---------------------------------------------------------------------"
echo "SECURITY WARNING:"
echo "This script will create/update a Cloud Build trigger using substitutions"
echo "potentially including sensitive values like GOOGLE_API_KEY and GITHUB_TOKEN"
echo "read from the '$ENV_FILE' file."
echo "For production environments, it is STRONGLY recommended to store"
echo "secrets in Google Secret Manager and reference them in the Cloud Build"
echo "trigger configuration instead of passing them as direct substitutions."
echo "See: https://cloud.google.com/build/docs/securing-builds/use-secrets"
echo "---------------------------------------------------------------------"
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborting."
    exit 1
fi

echo "Checking GitHub connection..."
gcloud alpha builds connections create github "$GITHUB_REPO_OWNER-$GITHUB_REPO_NAME-connection" --region="$REGION" --authorizer-token-secret-version=latest --app-installation-id=0 || echo "GitHub connection likely already exists or setup needs manual completion in browser."

# --- IMPORTANT: Ensure the Cloud Build GitHub App is installed... ---
echo "---------------------------------------------------------------------"
echo "IMPORTANT: Ensure the Cloud Build GitHub App is installed on the"
echo "'$GITHUB_REPO_OWNER/$GITHUB_REPO_NAME' repository or the '$GITHUB_REPO_OWNER' organization."
echo "You might have been prompted to install/authorize it in your browser"
echo "during the 'gcloud alpha builds connections create' step if it wasn't connected."
echo "Verify installation here: https://github.com/settings/installations"
echo "---------------------------------------------------------------------"
read -p "Press Enter to continue once the GitHub App is installed/verified..."

echo "Attempting to create/update Cloud Build trigger '$TRIGGER_NAME'..."
echo "Ignoring files: ${IGNORED_FILES}"
TRIGGER_DESCRIPTION="Trigger for ${GITHUB_REPO_NAME} ${TRIGGER_BRANCH} branch (Ignores Docs/Setup)" # Dynamic description

if gcloud beta builds triggers describe "$TRIGGER_NAME" --region="$REGION" > /dev/null 2>&1; then
    echo "Trigger '$TRIGGER_NAME' already exists. Updating it..."
    COMMAND_ARGS=(
        "beta" "builds" "triggers" "update" "github" "$TRIGGER_NAME"
        "--region=$REGION"
        "--repo-owner=$GITHUB_REPO_OWNER"
        "--repo-name=$GITHUB_REPO_NAME"
        "--branch-pattern=$TRIGGER_BRANCH"
        "--build-config=$BUILD_CONFIG_FILE"
        "--ignored-files=$IGNORED_FILES"
        "--description=$TRIGGER_DESCRIPTION"
        "--project=$GCP_PROJECT_ID"
    )
    if [ -n "$SUBSTITUTIONS" ]; then
        COMMAND_ARGS+=("--update-substitutions=$SUBSTITUTIONS")
    else
        echo "No substitutions provided for update."
    fi
    gcloud "${COMMAND_ARGS[@]}"

else
    echo "Trigger '$TRIGGER_NAME' does not exist. Creating it..."
     COMMAND_ARGS=(
        "beta" "builds" "triggers" "create" "github"
        "--name=$TRIGGER_NAME"
        "--region=$REGION"
        "--repo-owner=$GITHUB_REPO_OWNER"
        "--repo-name=$GITHUB_REPO_NAME"
        "--branch-pattern=$TRIGGER_BRANCH"
        "--build-config=$BUILD_CONFIG_FILE"
        "--ignored-files=$IGNORED_FILES"
        "--description=$TRIGGER_DESCRIPTION"
        "--project=$GCP_PROJECT_ID"
    )
    if [ -n "$SUBSTITUTIONS" ]; then
         COMMAND_ARGS+=("--substitutions=$SUBSTITUTIONS")
    else
        echo "No substitutions provided for create."
    fi
    gcloud "${COMMAND_ARGS[@]}"
fi

echo "Cloud Build trigger '$TRIGGER_NAME' setup complete."
echo "It will now trigger a build when changes are pushed to the '$TRIGGER_BRANCH' branch of '$GITHUB_REPO_OWNER/$GITHUB_REPO_NAME',"
echo "unless ALL changed files match the ignored patterns: ${IGNORED_FILES}"