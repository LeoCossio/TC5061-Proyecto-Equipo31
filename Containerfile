
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_TRACKING_URI=http://host.containers.internal:5000

COPY . /app

ENV MIBICI_MODEL_P1_PATH=/app/modelos_serving

RUN pip install --no-cache-dir \
    "mlflow==3.17.2" \
    "scikit-learn==1.9.1" \
    "pandas==2.3.2" \
    "numpy==2.3.3" \
    "fastapi==0.162.0" \
    "uvicorn==0.54.0" \
    "loguru==0.7.3" \
    "python-dotenv==1.2.2" \
    "typer==0.25.1" \
    && pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["uvicorn", "mibici.api:app", "--host", "0.0.0.0", "--port", "8000"]
