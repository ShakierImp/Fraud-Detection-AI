# docker/streamlit/Dockerfile
# ------------------------------------------------------------
# Streamlit frontend Dockerfile for Fraud Detection project
# ------------------------------------------------------------

# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment vars for Python
ENV PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# You can pin versions in a separate requirements file if needed
RUN pip install --no-cache-dir streamlit pandas requests

# Create non-root user
RUN useradd -ms /bin/bash appuser

# Create app directory
WORKDIR /app

# Copy source code
COPY ../../src ./src

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "src/frontend/dashboard_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ------------------------------------------------------------
# Example build and run:
#   docker build -t fraud-streamlit ./docker/streamlit
#   docker run -p 8501:8501 fraud-streamlit
# ------------------------------------------------------------
