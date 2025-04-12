#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Load .env file if it exists ---
if [ -f .env ]; then
  echo "Loading environment variables from .env file..."
  set -a
  source .env
  set +a
else
  echo "Warning: .env file not found. Relying on environment variables set externally."
fi
# --- End loading .env file ---

# --- Check GOOGLE_API_KEY ---
if [ -z "$GOOGLE_API_KEY" ]; then
  echo "Error: GOOGLE_API_KEY environment variable is not set."
  echo "Please set it in the .env file or export it externally."
  exit 1
fi
# --- End Check ---

# --- Define and Check GCP_PROJECT_ID ---
if [ -z "$GCP_PROJECT_ID" ]; then
  echo "Error: GCP_PROJECT_ID environment variable is not set."
  echo "Please set it in the .env file or export it externally."
  exit 1
fi
echo "Using GCP Project ID: $GCP_PROJECT_ID"
# --- End Check ---

# Some default values
REGION="us-central1"
REPO_ID="adk-agent-images"
SERVICE_NAME="adk-multi-agent-demo"
IMAGE_TAG="latest"
IMAGE_URI="${REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${REPO_ID}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "Starting Cloud Build..."
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
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY}"

echo "Deployment command finished."