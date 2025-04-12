FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip show google-adk

COPY . .

EXPOSE 8080

CMD streamlit run ui/app.py --server.port=${PORT} --server.headless=true