FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV PYTHONPATH=/app
ENV USE_STUB_PREDICTOR=1

EXPOSE 8000

CMD uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
