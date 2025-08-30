# docker/Dockerfile
# ------------------------------------------------------------
# Dockerfile for Fraud Detection FastAPI application
# Optimized for production with security and performance in mind
# ------------------------------------------------------------

# 1. Use a lightweight Python base image
FROM python:3.10-slim AS base

# 2. Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH"

# 3. Install system dependencies (security updates + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Create a non-root user for security
RUN useradd -m -u 1000 appuser

# 5. Set work directory
WORKDIR /app

# 6. Install Python dependencies first (optimized caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile -r requirements.txt

# 7. Copy the application source code
COPY src/ ./src/

# 8. Set ownership and permissions
RUN chown -R appuser:appuser /app
USER appuser

# 9. Expose the FastAPI port
EXPOSE 8000

# 10. Define a volume for model persistence
VOLUME ["/app/models"]

# 11. Healthcheck (verifies that the API is responding)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 12. Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "src.api.inference_api_fast:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------
# Example Build and Run:
#
# Build:
#   docker build -t fraud-detection-api -f docker/Dockerfile .
#
# Run (with models volume mounted):
#   docker run -d -p 8000:8000 \
#       -v $(pwd)/models:/app/models \
#       --name fraud-api \
#       fraud-detection-api
#
# Test health:
#   curl http://localhost:8000/health
# ------------------------------------------------------------
