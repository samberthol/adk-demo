#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load .env file if it exists ---
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  set -a # Automatically export all variables
  source .env
  set +a
else
  echo "Warning: .env file not found. Relying on environment variables set externally."
fi
# --- End loading .env file ---

# --- Check Required Environment Variables ---
REQUIRED_VARS=("GOOGLE_API_KEY" "GCP_PROJECT_ID" "REGION" "REPO_ID" "SERVICE_NAME" "IMAGE_TAG")
MISSING_VARS=()
for VAR_NAME in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!VAR_NAME}" ]; then # Use indirect expansion to check variable
    MISSING_VARS+=("$VAR_NAME")
  fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
  echo "Error: The following required environment variables are not set:"
  for VAR_NAME in "${MISSING_VARS[@]}"; do
    echo "  - $VAR_NAME"
  done
  echo "Please set them in the .env file or export them externally."
  exit 1
fi
# --- End Check ---

echo "Using GCP Project ID: $GCP_PROJECT_ID"
echo "Using Region: $REGION"
echo "Using Repository ID: $REPO_ID"
echo "Using Service Name: $SERVICE_NAME\n"

# Construct IMAGE_URI using environment variables
IMAGE_URI="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_ID}/${SERVICE_NAME}:${IMAGE_TAG}\n"
echo "Using Image URI: $IMAGE_URI"


echo "Starting Cloud Build..."
# Pass variables from environment to Cloud Build substitutions
gcloud builds submit . --config=cloudbuild.yaml \
  --project=${GCP_PROJECT_ID} \
  --substitutions=_REGION=${REGION},_REPO_ID=${REPO_ID},_SERVICE_NAME=${SERVICE_NAME},_IMAGE_TAG=${IMAGE_TAG}

echo "Build successful, starting Cloud Run deployment..."

gcloud run deploy ${SERVICE_NAME} \
  --image=${IMAGE_URI} \
  --platform=managed \
  --region=${REGION} \
  --port=8080 \
  --allow-unauthenticated \
  --project=${GCP_PROJECT_ID} \
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY}" # Pass API key securely

echo "Deployment command finished."