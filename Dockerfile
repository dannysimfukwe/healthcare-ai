FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV OLLAMA_BASE_URL=https://ai-api.42helv.com/v1
ENV DB_HOST=db
ENV DB_PORT=5432
ENV RAG_MODE=true

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sqlalchemy \
    psycopg2-binary \
    pgvector \
    openai \
    llama-index \
    pydantic \
    python-dotenv \
    httpx

COPY app.py .
COPY requirements.txt .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]