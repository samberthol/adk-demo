# Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn ffmpeg-python google-generativeai

COPY . .

EXPOSE 8080 8000

ENTRYPOINT ["/usr/bin/tini", "--"]