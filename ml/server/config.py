"""Server configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """ML sidecar server configuration."""

    # Server
    GRPC_PORT: int = int(os.getenv("GRPC_PORT", "50051"))
    HTTP_PORT: int = int(os.getenv("HTTP_PORT", "8081"))

    # Model paths
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", str(Path(__file__).parent.parent / "models")))

    # Inference
    DISRUPTION_THRESHOLD: float = float(os.getenv("DISRUPTION_THRESHOLD", "0.6"))
    ANOMALY_THRESHOLD: float = float(os.getenv("ANOMALY_THRESHOLD", "0.7"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Feature flags
    ENABLE_DISRUPTION: bool = os.getenv("ENABLE_DISRUPTION", "true").lower() == "true"
    ENABLE_ETA: bool = os.getenv("ENABLE_ETA", "true").lower() == "true"
    ENABLE_ANOMALY: bool = os.getenv("ENABLE_ANOMALY", "true").lower() == "true"

    @classmethod
    def model_path(cls, model_name: str, version: int | None = None) -> Path:
        """Get path to a specific model version."""
        base = cls.MODEL_DIR / model_name
        if version:
            return base / f"v{version}"
        # Find latest
        versions = sorted(
            [d for d in base.iterdir() if d.is_dir() and d.name.startswith("v")],
            key=lambda d: int(d.name[1:])
        ) if base.exists() else []
        if not versions:
            raise FileNotFoundError(f"No model found: {model_name}")
        return versions[-1]


config = Config()
