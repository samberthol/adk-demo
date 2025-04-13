# Dockerfile (Corrected pip install)
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# - ffmpeg is needed for audio transcoding in the FastAPI app
# - git is sometimes needed for pip installs from repos
# - tini is a simple init system to handle zombie processes and signals
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This now includes requests, gunicorn, ffmpeg-python should also be added if needed by api.py
# Assuming gunicorn needed for start.sh, ffmpeg-python for api.py
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn ffmpeg-python

# Copy the rest of the application code into the container
COPY . .

# Copy the startup script and make it executable
COPY start.sh .
RUN chmod +x start.sh

# Make port 8080 available to the world outside this container (Streamlit default)
# Also exposing 8000 for FastAPI internal communication if needed, though Cloud Run primarily uses 8080
EXPOSE 8080 8000

# Use tini as the entrypoint to properly handle signals and zombie processes
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run start.sh when the container launches
# This script will start both Gunicorn (FastAPI) and Streamlit
CMD ["./start.sh"]