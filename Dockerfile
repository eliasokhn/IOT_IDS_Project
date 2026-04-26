# ────────────────────────────────────────────────────────────────
# Dockerfile — IoT Intrusion Detection System
# ────────────────────────────────────────────────────────────────
# Build:  docker build -t iot-ids:latest .
# Run:    docker run -p 8000:8000 iot-ids:latest
# Docs:   http://localhost:8000/docs
# ────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer — rebuilt only when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 5 -r requirements.txt

# Source code and configuration
COPY src/ ./src/
COPY configs/ ./configs/
COPY run_pipeline.py .

# Pre-trained model artifacts (6 models + preprocessor + class name files)
# These are checked into git so they're always present at build time.
COPY models/ ./models/

# Evaluation reports (metrics JSON + plots) — useful for reference
COPY reports/ ./reports/

# Create a non-root user for security
RUN useradd -m -u 1000 idsuser && chown -R idsuser:idsuser /app
USER idsuser

# Environment — override at runtime with -e flags
ENV ARTIFACTS_DIR=models
ENV MODEL_NAME=gb
ENV TASK=binary
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
