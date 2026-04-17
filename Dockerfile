# ────────────────────────────────────────────────────────────────
# Dockerfile — IoT Intrusion Detection API
# ────────────────────────────────────────────────────────────────
# Build:  docker build -t yourteam/iot-ids:latest .
# Run:    docker run -p 8000:8000 yourteam/iot-ids:latest
# Push:   docker push yourteam/iot-ids:latest
# ────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Copy trained model artifacts (must be built before docker build)
# If models/ directory is empty, the API runs in demo mode
COPY models/ ./models/

# Copy any saved reports (optional — for reference)
# COPY reports/ ./reports/

# Create non-root user for security
RUN useradd -m -u 1000 idsuser && chown -R idsuser:idsuser /app
USER idsuser

# Environment variables (can be overridden at runtime)
ENV ARTIFACTS_DIR=models
ENV MODEL_NAME=gb
ENV TASK=binary
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
