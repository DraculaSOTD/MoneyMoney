# Multi-Model Cryptocurrency ML Trading System
# Production-ready Docker container

FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/historical saved_models config && \
    chown -R app:app /app

# Switch to app user
USER app

# Expose port for API (if enabled)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "main.py"]

# -----------------------------------------------------------------------------
# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black flake8 mypy jupyter

# Install debugging tools
RUN pip install --no-cache-dir ipdb pdb-tools

USER app

# Override command for development
CMD ["python", "main.py", "--debug"]

# -----------------------------------------------------------------------------
# Production stage
FROM base as production

# Copy optimized configuration
COPY config/production_config.yaml config/system_config.yaml

# Set production environment
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Production command with monitoring
CMD ["python", "-u", "main.py"]