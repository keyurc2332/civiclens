# CivicLens application image -- used for both the API and dashboard
# services in docker-compose.yml (same image, different commands).
FROM python:3.10-slim

WORKDIR /app

# System deps: libgomp1 is required by xgboost at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY sql/ sql/
COPY src/ src/
COPY scripts/ scripts/
COPY .streamlit/ .streamlit/

# Default command is the API; the dashboard service overrides this
# in docker-compose.yml.
EXPOSE 8000 8501
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]