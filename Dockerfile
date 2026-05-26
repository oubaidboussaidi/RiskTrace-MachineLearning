# ──────────────────────────────────────────────────────────────
#  RiskTrace — ML Service (FastAPI + Isolation Forest)
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained models
COPY src ./src
COPY models ./models

EXPOSE 8000

# Run the FastAPI app via Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
