FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY proto/ proto/
COPY server/ server/
COPY pipelines/ pipelines/
COPY utils/ utils/
COPY models/ models/

# Generate gRPC stubs
RUN python -m grpc_tools.protoc \
    -I proto/ \
    --python_out=server/ \
    --grpc_python_out=server/ \
    proto/inference.proto

# Expose ports
EXPOSE 50051 8081

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8081/health'); exit(0 if r.ok else 1)"

# Run
CMD ["python", "-m", "server.app"]
