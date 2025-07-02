# Dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tini \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/google/adk-samples.git /tmp/adk-samples && \
    mkdir -p /app/agents/llm_auditor && \
    cp -R /tmp/adk-samples/python/agents/llm-auditor/. /app/agents/llm_auditor/ && \
    rm -rf /tmp/adk-samples

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

EXPOSE 8080 8000

ENTRYPOINT ["/usr/bin/tini", "--"]