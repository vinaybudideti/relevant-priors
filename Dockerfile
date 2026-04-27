FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/ artifacts/
COPY data/ data/

ENV PYTHONPATH=/app
ENV ARTIFACTS_DIR=/app/artifacts
ENV USE_STUB_PREDICTOR=0

EXPOSE 8000

CMD uvicorn src.app.main:app --host 0.0.0.0 --port ${PORT:-8000}
