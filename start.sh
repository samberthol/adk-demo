#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Start the FastAPI app using Gunicorn in the background
# Listen on port 8000 (adjust if needed)
# Use Uvicorn workers for FastAPI
echo "Starting FastAPI server on port 8000..."
gunicorn ui.api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 &

# Start the Streamlit app
# Streamlit respects the PORT environment variable set by Cloud Run (defaults to 8080)
echo "Starting Streamlit server on port $PORT (or default 8080)..."
streamlit run ui/app.py --server.port ${PORT:-8080} --server.address 0.0.0.0 --server.headless true

# Keep the script running - Streamlit will run in the foreground
wait -n
exit $?