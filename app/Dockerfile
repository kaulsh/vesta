# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy the vesta_ml package
COPY packages/vesta_ml /app/packages/vesta_ml

# Copy application code (as a module)
COPY app /app/app

# Install vesta_ml package
RUN pip install --no-cache-dir -e /app/packages/vesta_ml

# Create instance directory for SQLite database
RUN mkdir -p /app/app/instance

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app:create_app
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Use gunicorn for production
RUN pip install --no-cache-dir gunicorn

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()"

# Run the application with gunicorn (with SSL support if certificates provided)
CMD if [ -f "${SSL_CERT:-}" ] && [ -f "${SSL_KEY:-}" ]; then \
      gunicorn --bind 0.0.0.0:8000 \
               --certfile="${SSL_CERT}" \
               --keyfile="${SSL_KEY}" \
               --workers 2 --threads 2 --timeout 60 \
               "app:create_app()"; \
    else \
      gunicorn --bind 0.0.0.0:8000 \
               --workers 2 --threads 2 --timeout 60 \
               "app:create_app()"; \
    fi
