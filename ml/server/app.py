"""
NexusIQ ML Sidecar — main entry point.
Runs both gRPC (for Go backend calls) and HTTP (for health checks) servers.
"""

import threading
import uvicorn
from fastapi import FastAPI

from server.config import config
from server.health import router as health_router
from server.inference import registry
from server.grpc_server import serve as start_grpc
from utils.logger import setup_logger

logger = setup_logger("nexusiq.ml.server", config.LOG_LEVEL)


def create_app() -> FastAPI:
    """Create FastAPI app for HTTP health endpoints."""
    app = FastAPI(
        title="NexusIQ ML Sidecar",
        description="ML inference service for NexusIQ supply chain intelligence",
        version="0.1.0",
    )
    app.include_router(health_router)
    return app


def main():
    """Start both gRPC and HTTP servers."""
    # Load models
    logger.info("loading_models")
    registry.load_all()

    # Start gRPC server in background thread
    grpc_server = start_grpc()
    logger.info("grpc_ready", port=config.GRPC_PORT)

    # Start HTTP server (blocking) for health checks
    app = create_app()
    logger.info("http_starting", port=config.HTTP_PORT)
    uvicorn.run(app, host="0.0.0.0", port=config.HTTP_PORT, log_level="warning")


if __name__ == "__main__":
    main()
