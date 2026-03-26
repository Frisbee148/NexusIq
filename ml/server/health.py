"""HTTP health check endpoint (for k8s liveness/readiness probes)."""

from fastapi import APIRouter
from server.inference import registry

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness probe — is the server running?"""
    return {"status": "ok", "uptime": registry.uptime_seconds}


@router.get("/ready")
async def readiness():
    """Readiness probe — are models loaded and ready to serve?"""
    models_status = {
        "disruption": registry.disruption_model is not None,
        "eta": registry.eta_model is not None,
        "anomaly": registry.anomaly_model is not None,
        "risk_scorer": True,
    }
    all_ready = all(models_status.values())
    return {
        "ready": all_ready,
        "models": models_status,
        "uptime": registry.uptime_seconds,
    }
