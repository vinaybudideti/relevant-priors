FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

# Train artifacts inside the container so they match the runtime
RUN python -m src.app.train --input data/relevant_priors_public.json --out artifacts/

ENV PYTHONPATH=/app
ENV ARTIFACTS_DIR=/app/artifacts
ENV USE_STUB_PREDICTOR=0

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
