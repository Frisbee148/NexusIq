# NexusIQ ML Pipeline — Block 1 & Block 2 Setup Prompt

> **Use this file as a prompt in Claude Code.** It contains everything needed to initialise the ML project structure, Python environment, gRPC contract, FastAPI sidecar skeleton, Docker setup, and Makefile automation. Follow every instruction precisely.

---

## PROJECT CONTEXT

**Product:** NexusIQ — India's Multimodal Supply Chain Intelligence Platform  
**What we're building right now:** The ML inference sidecar — a Python microservice that trains and serves 4 ML models (Disruption Predictor, ETA Confidence Bands, Anomaly Detector, Supplier Risk Scorer) via gRPC to a Go backend.  
**Monorepo structure:** This `ml/` directory sits inside a larger monorepo. The Go backend and Next.js frontend are siblings at the root level. We are ONLY building the `ml/` portion right now.

---

## BLOCK 1: PROJECT STRUCTURE & ENVIRONMENT

### Task 1.1 — Create the full ML directory tree

Create the following directory structure under `ml/` at the project root. Create all directories and placeholder files exactly as specified. Every `__init__.py` should be an empty file. Every placeholder `.py` file should have a module docstring explaining its purpose (one line) and a `# TODO: implement` comment.

```
ml/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.ml.yaml
├── Makefile
├── .env.example
├── .gitignore
├── setup.cfg
│
├── proto/
│   └── inference.proto
│
├── data/
│   ├── .gitkeep
│   ├── raw/
│   │   └── .gitkeep
│   ├── clean/
│   │   └── .gitkeep
│   ├── features/
│   │   └── .gitkeep
│   └── synthetic/
│       └── .gitkeep
│
├── scripts/
│   ├── __init__.py
│   ├── download_imd_weather.py
│   ├── download_openweather.py
│   ├── download_port_data.py
│   ├── download_news.py
│   ├── generate_shipments.py
│   ├── generate_gps_telemetry.py
│   ├── generate_port_events.py
│   ├── generate_weather_disruptions.py
│   ├── generate_suppliers.py
│   └── seed_neo4j_graph.py
│
├── pipelines/
│   ├── __init__.py
│   ├── base.py
│   ├── disruption/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── config.yaml
│   ├── eta/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── config.yaml
│   ├── anomaly/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── config.yaml
│   └── risk/
│       ├── __init__.py
│       └── scorer.py
│
├── models/
│   ├── .gitkeep
│   ├── disruption/
│   │   └── .gitkeep
│   ├── eta/
│   │   └── .gitkeep
│   └── anomaly/
│       └── .gitkeep
│
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── grpc_server.py
│   ├── inference.py
│   ├── health.py
│   └── config.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── geo.py
│   ├── io.py
│   ├── metrics.py
│   └── constants.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_disruption_pipeline.py
    ├── test_eta_pipeline.py
    ├── test_anomaly_pipeline.py
    ├── test_risk_scorer.py
    ├── test_grpc_server.py
    └── test_inference.py
```

### Task 1.2 — Write `requirements.txt`

Pin every dependency to an exact version. These are the ONLY dependencies needed for Block 1 & 2. Do NOT add tensorflow, torch, or any deep learning library.

```txt
# ── Core ML ──
xgboost==2.1.3
scikit-learn==1.6.1
pandas==2.2.3
numpy==1.26.4
optuna==4.1.0
joblib==1.4.2

# ── Data formats ──
pyarrow==18.1.0
pyyaml==6.0.2
python-dotenv==1.0.1

# ── API & Data Acquisition ──
requests==2.32.3
beautifulsoup4==4.12.3

# ── Geospatial ──
geopy==2.4.1
shapely==2.0.6

# ── gRPC ──
grpcio==1.68.1
grpcio-tools==1.68.1
grpcio-health-checking==1.68.1
protobuf==5.29.2

# ── Server ──
fastapi==0.115.6
uvicorn[standard]==0.34.0

# ── Evaluation & Plotting ──
matplotlib==3.10.0
seaborn==0.13.2

# ── Logging ──
structlog==24.4.0
```

### Task 1.3 — Write `requirements-dev.txt`

```txt
-r requirements.txt
pytest==8.3.4
pytest-asyncio==0.24.0
pytest-cov==6.0.0
ruff==0.8.4
mypy==1.13.0
```

### Task 1.4 — Write `ml/.gitignore`

```gitignore
# Data (never commit raw data or model artifacts)
data/raw/*
data/clean/*
data/features/*
data/synthetic/*
!data/**/.gitkeep

# Trained models
models/disruption/v*/
models/eta/v*/
models/anomaly/v*/
!models/**/.gitkeep

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environment
venv/
.venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment
.env

# Generated gRPC stubs
server/*_pb2.py
server/*_pb2_grpc.py

# OS
.DS_Store
Thumbs.db

# Jupyter (if used for exploration)
.ipynb_checkpoints/
```

### Task 1.5 — Write `ml/.env.example`

```env
# ── API Keys ──
OPENWEATHER_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here

# ── Database ──
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=nexusiq
POSTGRES_USER=nexusiq
POSTGRES_PASSWORD=nexusiq_dev

# ── Redis ──
REDIS_HOST=localhost
REDIS_PORT=6379

# ── Neo4j ──
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=nexusiq_dev

# ── Kafka ──
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# ── ML Server ──
GRPC_PORT=50051
HTTP_PORT=8081
MODEL_DIR=./models
LOG_LEVEL=INFO

# ── Training ──
OPTUNA_N_TRIALS=50
RANDOM_SEED=42
```

### Task 1.6 — Write `ml/setup.cfg`

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short

[ruff]
line-length = 100
target-version = "py311"

[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
ignore_missing_imports = True
```

---

## BLOCK 1 CONTINUED: CORE UTILITY MODULES

### Task 1.7 — Write `ml/utils/constants.py`

This file defines ALL the domain constants used across the ML pipeline. These are India-specific logistics constants that calibrate our synthetic data and models.

```python
"""
NexusIQ domain constants — India logistics network parameters.
All distances in km, costs in INR, times in hours, carbon in kg CO2 per tonne-km.
"""

RANDOM_SEED = 42

# ── Indian Transport Modes ──
TRANSPORT_MODES = ["road", "rail", "air", "sea", "waterway"]

# ── Average speed by mode (km/h) — Indian conditions ──
MODE_SPEED = {
    "road": 35,        # NH average including stops, tolls (not highway max)
    "rail": 25,        # freight rail average (not passenger)
    "rail_dfc": 50,    # Dedicated Freight Corridor speed
    "air": 500,        # effective including ground handling
    "sea": 25,         # knots converted, coastal shipping
    "waterway": 10,    # NW-1 Ganga average
}

# ── Cost per tonne-km (INR) — Indian benchmarks ──
MODE_COST_PER_TONNE_KM = {
    "road": 2.50,
    "rail": 1.20,
    "rail_dfc": 1.00,
    "air": 18.00,
    "sea": 0.60,
    "waterway": 0.80,
}

# ── Carbon emission per tonne-km (kg CO2) ──
MODE_CARBON_PER_TONNE_KM = {
    "road": 0.062,
    "rail": 0.022,
    "rail_dfc": 0.018,
    "air": 0.602,
    "sea": 0.008,
    "waterway": 0.016,
}

# ── Mode transfer time (hours) — time to switch between modes at a hub ──
MODE_TRANSFER_TIME = {
    ("road", "rail"): 4,
    ("road", "air"): 6,
    ("road", "sea"): 8,
    ("road", "waterway"): 3,
    ("rail", "sea"): 6,
    ("rail", "road"): 4,
    ("sea", "road"): 8,
    ("sea", "rail"): 6,
    ("air", "road"): 4,
    ("waterway", "road"): 3,
}

# ── Seasons (Indian context) ──
SEASONS = {
    "summer": (3, 5),       # March–May
    "monsoon": (6, 9),      # June–September
    "post_monsoon": (10, 11), # October–November
    "winter": (12, 2),      # December–February
}

# ── Monsoon disruption multipliers (applied to transit time) ──
MONSOON_DISRUPTION_MULTIPLIER = {
    "road": 1.4,       # 40% slower
    "rail": 1.2,       # 20% slower
    "air": 1.1,        # minimal impact
    "sea": 1.3,        # port congestion
    "waterway": 1.6,   # depth issues, current changes
}

# ── Major Indian Logistics Hubs (for graph seeding) ──
# Format: (name, lat, lon, modes_available)
LOGISTICS_HUBS = [
    # Metro / Tier-1
    ("Mumbai", 19.0760, 72.8777, ["road", "rail", "air", "sea"]),
    ("Delhi", 28.7041, 77.1025, ["road", "rail", "air"]),
    ("Chennai", 13.0827, 80.2707, ["road", "rail", "air", "sea"]),
    ("Kolkata", 22.5726, 88.3639, ["road", "rail", "air", "sea", "waterway"]),
    ("Bengaluru", 12.9716, 77.5946, ["road", "rail", "air"]),
    ("Hyderabad", 17.3850, 78.4867, ["road", "rail", "air"]),
    ("Ahmedabad", 23.0225, 72.5714, ["road", "rail", "air"]),
    ("Pune", 18.5204, 73.8567, ["road", "rail", "air"]),
    # Key ports
    ("JNPT_Nhava_Sheva", 18.9500, 72.9500, ["sea", "road", "rail"]),
    ("Mundra", 22.8394, 69.7186, ["sea", "road", "rail"]),
    ("Visakhapatnam", 17.6868, 83.2185, ["sea", "road", "rail"]),
    ("Kandla", 23.0333, 70.2167, ["sea", "road", "rail"]),
    ("Cochin", 9.9312, 76.2673, ["sea", "road", "rail"]),
    ("Paradip", 20.3164, 86.6085, ["sea", "road", "rail"]),
    ("Tuticorin", 8.7642, 78.1348, ["sea", "road"]),
    ("Haldia", 22.0667, 88.0698, ["sea", "road", "rail", "waterway"]),
    # DFC Corridor hubs
    ("Dadri", 28.5535, 77.5552, ["rail", "rail_dfc", "road"]),
    ("Rewari", 28.1970, 76.6190, ["rail", "rail_dfc", "road"]),
    ("Palanpur", 24.1725, 72.4340, ["rail", "rail_dfc", "road"]),
    ("Ludhiana", 30.9010, 75.8573, ["road", "rail", "rail_dfc"]),
    ("Dankuni", 22.6800, 88.2900, ["rail", "rail_dfc", "road", "waterway"]),
    ("Khurja", 28.2500, 77.8500, ["rail", "rail_dfc", "road"]),
    ("Sonnagar", 24.8800, 83.8700, ["rail", "rail_dfc", "road"]),
    # IWT / Waterway hubs
    ("Varanasi", 25.3176, 82.9739, ["road", "rail", "waterway"]),
    ("Patna", 25.6093, 85.1376, ["road", "rail", "waterway"]),
    ("Sahibganj", 25.2464, 87.6367, ["road", "waterway"]),
    ("Guwahati", 26.1445, 91.7362, ["road", "rail", "waterway"]),
    # Air cargo hubs
    ("Hyderabad_Airport", 17.2403, 78.4294, ["air", "road"]),
    ("Bengaluru_Airport", 13.1986, 77.7066, ["air", "road"]),
    ("Delhi_Airport", 28.5562, 77.1000, ["air", "road"]),
    ("Mumbai_Airport", 19.0896, 72.8656, ["air", "road"]),
    # Industrial clusters
    ("Surat", 21.1702, 72.8311, ["road", "rail"]),
    ("Jamshedpur", 22.8046, 86.2029, ["road", "rail"]),
    ("Nagpur", 21.1458, 79.0882, ["road", "rail", "air"]),
    ("Coimbatore", 11.0168, 76.9558, ["road", "rail", "air"]),
    ("Jaipur", 26.9124, 75.7873, ["road", "rail", "air"]),
    ("Lucknow", 26.8467, 80.9462, ["road", "rail", "air"]),
    ("Kanpur", 26.4499, 80.3319, ["road", "rail"]),
    ("Indore", 22.7196, 75.8577, ["road", "rail", "air"]),
    ("Bhopal", 23.2599, 77.4126, ["road", "rail", "air"]),
    ("Goa_Mormugao", 15.4127, 73.8007, ["sea", "road", "rail"]),
    ("Mangalore", 12.9141, 74.8560, ["sea", "road", "rail"]),
    ("Raipur", 21.2514, 81.6296, ["road", "rail", "air"]),
    ("Dhanbad", 23.7957, 86.4304, ["road", "rail"]),
    # Border / International corridor
    ("Petrapole", 23.1800, 88.8700, ["road"]),  # India-Bangladesh border
    ("Attari_Wagah", 31.6050, 74.5700, ["road", "rail"]),  # India-Pakistan border
    ("Birgunj", 27.0100, 84.8800, ["road", "rail"]),  # India-Nepal border
]

# ── Cargo Types ──
CARGO_TYPES = [
    "general",
    "containers",
    "bulk_dry",        # coal, iron ore, cement
    "bulk_liquid",     # petroleum, chemicals
    "perishables",     # fruits, vegetables, dairy
    "pharma",          # temperature sensitive
    "automobiles",
    "electronics",
    "textiles",
    "fmcg",
    "hazardous",
    "oversized",
]

# ── Disruption Types ──
DISRUPTION_TYPES = [
    "weather_flood",
    "weather_cyclone",
    "weather_heavy_rain",
    "weather_fog",
    "strike_transport",
    "strike_port",
    "strike_general",
    "port_congestion",
    "rail_disruption",
    "road_accident",
    "policy_change",
    "customs_delay",
    "waterway_low_depth",
]

# ── Disruption Severity Levels ──
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# ── Risk Score Thresholds ──
RISK_THRESHOLDS = {
    "GREEN": (0, 40),
    "AMBER": (41, 70),
    "RED": (71, 100),
}
```

### Task 1.8 — Write `ml/utils/logger.py`

```python
"""Structured logging configuration for NexusIQ ML pipeline."""

import structlog
import logging
import sys


def setup_logger(name: str = "nexusiq.ml", level: str = "INFO") -> structlog.BoundLogger:
    """Configure and return a structured logger instance."""

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)


# Default logger instance
logger = setup_logger()
```

### Task 1.9 — Write `ml/utils/io.py`

```python
"""I/O utilities for data loading, saving, and model artifact management."""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import joblib
import yaml

from utils.logger import logger


# ── Base paths ──
ML_ROOT = Path(__file__).parent.parent
DATA_DIR = ML_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = ML_ROOT / "models"


def ensure_dirs() -> None:
    """Create all data directories if they don't exist."""
    for d in [RAW_DIR, CLEAN_DIR, FEATURES_DIR, SYNTHETIC_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    logger.info("loading_csv", path=str(path))
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("saved_csv", path=str(path), rows=len(df))


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    logger.info("loading_parquet", path=str(path))
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet format (compact, fast, typed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("saved_parquet", path=str(path), rows=len(df))


def load_yaml(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_model(model: Any, model_name: str, version: int, metrics: dict,
               hyperparams: dict, dataset_path: Path | None = None) -> Path:
    """
    Save a trained model artifact with full metadata for reproducibility.

    Creates:
      models/{model_name}/v{version}/
        ├── model.pkl          # serialised model
        └── metadata.json      # training context
    """
    model_dir = MODELS_DIR / model_name / f"v{version}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    # Compute dataset hash for reproducibility tracking
    dataset_hash = None
    if dataset_path and dataset_path.exists():
        dataset_hash = hashlib.md5(dataset_path.read_bytes()).hexdigest()[:12]

    # Save metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "hyperparams": hyperparams,
        "dataset_hash": dataset_hash,
        "model_file": "model.pkl",
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("saved_model",
                model=model_name,
                version=version,
                metrics=metrics,
                path=str(model_dir))
    return model_dir


def load_model(model_name: str, version: int | None = None) -> tuple[Any, dict]:
    """
    Load a trained model and its metadata.
    If version is None, loads the latest version.
    """
    model_base = MODELS_DIR / model_name

    if version is None:
        # Find latest version
        versions = sorted(
            [d for d in model_base.iterdir() if d.is_dir() and d.name.startswith("v")],
            key=lambda d: int(d.name[1:])
        )
        if not versions:
            raise FileNotFoundError(f"No trained models found for {model_name}")
        model_dir = versions[-1]
    else:
        model_dir = model_base / f"v{version}"

    model = joblib.load(model_dir / "model.pkl")
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info("loaded_model",
                model=model_name,
                version=metadata["version"],
                metrics=metadata["metrics"])
    return model, metadata
```

### Task 1.10 — Write `ml/utils/geo.py`

```python
"""Geospatial utilities for distance calculation and coordinate operations."""

import math
from typing import Tuple

from geopy.distance import geodesic


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in kilometres."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def geodesic_km(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Precise geodesic distance using WGS-84 ellipsoid. Slower but more accurate."""
    return geodesic(coord1, coord2).kilometers


def get_season(month: int) -> str:
    """Return Indian season name for a given month (1-12)."""
    if month in (3, 4, 5):
        return "summer"
    elif month in (6, 7, 8, 9):
        return "monsoon"
    elif month in (10, 11):
        return "post_monsoon"
    else:
        return "winter"


def is_coastal(lat: float, lon: float, coastal_buffer_km: float = 100) -> bool:
    """
    Rough check if a point is within coastal_buffer_km of the Indian coastline.
    Uses a simplified polygon — good enough for feature engineering.
    """
    # Simplified Indian coastline reference points
    coastal_refs = [
        (8.08, 77.55),    # Kanyakumari
        (9.97, 76.27),    # Cochin
        (12.91, 74.86),   # Mangalore
        (15.41, 73.80),   # Goa
        (18.95, 72.83),   # Mumbai
        (22.47, 70.06),   # Porbandar
        (23.03, 70.22),   # Kandla
        (22.84, 69.72),   # Mundra
        (13.08, 80.27),   # Chennai
        (17.69, 83.22),   # Visakhapatnam
        (20.32, 86.61),   # Paradip
        (22.07, 88.07),   # Haldia
        (22.57, 88.36),   # Kolkata
    ]
    return any(
        haversine_km(lat, lon, ref_lat, ref_lon) < coastal_buffer_km
        for ref_lat, ref_lon in coastal_refs
    )
```

### Task 1.11 — Write `ml/utils/metrics.py`

```python
"""Evaluation metrics and reporting utilities for NexusIQ ML models."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from utils.logger import logger


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray | None = None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray | None = None) -> dict:
    """Compute binary classification metrics with false positive rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
    }


def calibration_error(y_true: np.ndarray, y_pred_quantiles: dict[str, np.ndarray]) -> dict:
    """
    Compute calibration error for quantile predictions (ETA model).
    Checks if actual fraction within each quantile band matches the nominal coverage.
    """
    errors = {}
    for quantile_name, threshold in y_pred_quantiles.items():
        nominal = float(quantile_name.replace("p", "")) / 100.0
        actual_coverage = float(np.mean(y_true <= threshold))
        errors[quantile_name] = {
            "nominal": nominal,
            "actual": actual_coverage,
            "error": abs(nominal - actual_coverage),
        }
    avg_error = float(np.mean([v["error"] for v in errors.values()]))
    return {"per_quantile": errors, "mean_calibration_error": avg_error}


def save_eval_report(metrics: dict, model_name: str, version: int,
                     output_dir: Path | None = None) -> Path:
    """Save evaluation report as JSON."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "models" / model_name / f"v{version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("saved_eval_report", path=str(report_path))
    return report_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list[str], save_path: Path) -> None:
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_confusion_matrix", path=str(save_path))


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray,
                          save_path: Path) -> None:
    """Generate and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_pr_curve", path=str(save_path))
```

---

## BLOCK 2: gRPC CONTRACT & SERVER SKELETON

### Task 2.1 — Write `ml/proto/inference.proto`

This is the contract between the Go backend and Python ML sidecar. It must be written EXACTLY as specified — both sides generate code from this single file.

```protobuf
syntax = "proto3";

package nexusiq.ml;

option go_package = "github.com/nexusiq/nexusiq/pkg/mlpb";

// ─────────────────────────────────────────────
// NexusIQ ML Inference Service
// Called by Go backend via gRPC
// ─────────────────────────────────────────────

service InferenceService {
  // Predict disruption probability for a transport corridor
  rpc PredictDisruption(DisruptionRequest) returns (DisruptionResponse);

  // Predict delivery time with confidence bands (p50/p75/p90/p99)
  rpc PredictETA(ETARequest) returns (ETAResponse);

  // Score a shipment for anomalies (route deviation, weight mismatch, etc.)
  rpc ScoreAnomaly(AnomalyRequest) returns (AnomalyResponse);

  // Batch score multiple suppliers for risk
  rpc BatchScoreRisk(RiskBatchRequest) returns (RiskBatchResponse);

  // Health check — returns model loading status
  rpc HealthCheck(Empty) returns (HealthResponse);
}

// ── Common ──

message Empty {}

// ── Disruption Prediction ──

message DisruptionRequest {
  string corridor_id = 1;
  // Weather features
  float temperature_celsius = 2;
  float rainfall_mm = 3;
  float wind_speed_kmh = 4;
  float humidity_pct = 5;
  // Network state features
  float congestion_index = 6;         // 0.0–1.0 normalised
  float port_utilization_pct = 7;     // 0–100 for port corridors
  // NLP features
  float news_sentiment_score = 8;     // -1.0 to +1.0
  int32 strike_mention_count_24h = 9;
  // Context features
  string season = 10;                 // "summer" | "monsoon" | "post_monsoon" | "winter"
  int32 hour_of_day = 11;             // 0–23
  int32 day_of_week = 12;             // 0=Mon, 6=Sun
  bool is_dfc_corridor = 13;
  bool is_coastal_corridor = 14;
  string primary_mode = 15;           // "road" | "rail" | "sea" | "air" | "waterway"
  // Historical features
  float corridor_avg_delay_hours_30d = 16;
  float corridor_disruption_rate_90d = 17;  // fraction of days with disruption
}

message DisruptionResponse {
  float disruption_probability = 1;          // 0.0–1.0
  string predicted_disruption_type = 2;      // from DISRUPTION_TYPES
  string severity = 3;                       // "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
  repeated string contributing_factors = 4;  // top 3 features driving the prediction
  float model_confidence = 5;               // how confident the model is (0.0–1.0)
  string model_version = 6;                 // e.g. "disruption/v1"
}

// ── ETA Prediction ──

message ETARequest {
  string origin_hub = 1;
  string destination_hub = 2;
  repeated string mode_sequence = 3;         // ["road", "rail", "waterway"]
  string cargo_type = 4;                     // from CARGO_TYPES
  float cargo_weight_tonnes = 5;
  string dispatch_timestamp = 6;             // ISO-8601
  // Current conditions
  float route_disruption_score = 7;          // 0.0–1.0, from disruption model
  string season = 8;
  float total_distance_km = 9;
  int32 num_mode_transfers = 10;
  // Optional: carrier/lane history
  float carrier_ontime_rate_90d = 11;        // 0.0–1.0
  float lane_avg_transit_hours = 12;         // historical average for this OD pair
}

message ETAResponse {
  // Delivery time predictions in hours from dispatch
  float p50_hours = 1;
  float p75_hours = 2;
  float p90_hours = 3;
  float p99_hours = 4;
  // Derived timestamps (computed from dispatch_timestamp + hours)
  string p50_eta = 5;                        // ISO-8601 timestamp
  string p75_eta = 6;
  string p90_eta = 7;
  string p99_eta = 8;
  // Metadata
  float confidence_width_hours = 9;          // p99 - p50
  string model_version = 10;
}

// ── Anomaly Detection ──

message AnomalyRequest {
  string shipment_id = 1;
  // GPS track data
  repeated GPSPoint gps_track = 2;
  // Weight data
  float declared_weight_kg = 3;
  float measured_weight_kg = 4;              // at checkpoint
  // Timing
  float expected_transit_hours = 5;
  float actual_elapsed_hours = 6;
  // Route
  float planned_distance_km = 7;
  float actual_distance_km = 8;             // from GPS track
  // Context
  string cargo_type = 9;
  string transport_mode = 10;
}

message GPSPoint {
  double latitude = 1;
  double longitude = 2;
  string timestamp = 3;                     // ISO-8601
  float speed_kmh = 4;
  bool is_stopped = 5;                      // speed < 2 km/h for > 15 min
}

message AnomalyResponse {
  float anomaly_score = 1;                  // 0.0–1.0
  string anomaly_type = 2;                  // "route_deviation" | "stationary_alert" | "weight_mismatch" | "timing_anomaly" | "speed_anomaly" | "none"
  string explanation = 3;                   // human-readable description
  bool should_alert = 4;                    // true if score exceeds threshold
  string severity = 5;                      // "LOW" | "MEDIUM" | "HIGH"
  string model_version = 6;
}

// ── Supplier Risk Scoring ──

message SupplierRiskInput {
  string supplier_id = 1;
  // Location risk
  float location_flood_frequency = 2;       // events per year (historical)
  float location_cyclone_exposure = 3;      // 0.0–1.0
  bool is_coastal = 4;
  // Performance
  float ontime_delivery_rate_90d = 5;       // 0.0–1.0
  float avg_delay_hours = 6;
  int32 total_shipments_90d = 7;
  // Financial / reputation signals
  float news_sentiment_30d = 8;             // -1.0 to +1.0
  bool gst_compliant = 9;
  int32 negative_news_count_30d = 10;
  // Dependency
  int32 num_ports_used = 11;                // port concentration risk
  float single_port_dependency_pct = 12;    // % of shipments through one port
}

message SupplierRiskOutput {
  string supplier_id = 1;
  float risk_score = 2;                     // 0–100
  string risk_level = 3;                    // "GREEN" | "AMBER" | "RED"
  repeated string top_risk_factors = 4;     // top 3 contributors
  string trend = 5;                         // "improving" | "stable" | "worsening"
}

message RiskBatchRequest {
  repeated SupplierRiskInput suppliers = 1;
}

message RiskBatchResponse {
  repeated SupplierRiskOutput results = 1;
  string scorer_version = 2;
}

// ── Health Check ──

message HealthResponse {
  bool healthy = 1;
  bool disruption_model_loaded = 2;
  bool eta_model_loaded = 3;
  bool anomaly_model_loaded = 4;
  bool risk_scorer_ready = 5;
  string server_version = 6;
  int64 uptime_seconds = 7;
}
```

### Task 2.2 — Write `ml/server/config.py`

```python
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
```

### Task 2.3 — Write `ml/server/inference.py`

This is the model loading and prediction logic. It loads all trained models at startup and exposes prediction methods that the gRPC server calls.

```python
"""Model loading and inference logic for all NexusIQ ML models."""

import time
from pathlib import Path
from typing import Any

import numpy as np
import joblib

from server.config import config
from utils.logger import logger
from utils.constants import RISK_THRESHOLDS, SEVERITY_LEVELS


class ModelRegistry:
    """
    Loads and manages all ML model artifacts.
    Initialised once at server startup, shared across gRPC calls.
    """

    def __init__(self):
        self.disruption_model: Any | None = None
        self.eta_model: Any | None = None
        self.anomaly_model: Any | None = None
        self._loaded_versions: dict[str, str] = {}
        self._start_time = time.time()

    def load_all(self) -> None:
        """Load all available models. Skip gracefully if a model isn't trained yet."""
        if config.ENABLE_DISRUPTION:
            self._load_model("disruption")
        if config.ENABLE_ETA:
            self._load_model("eta")
        if config.ENABLE_ANOMALY:
            self._load_model("anomaly")

    def _load_model(self, name: str) -> None:
        """Load a single model by name."""
        try:
            model_dir = config.model_path(name)
            model = joblib.load(model_dir / "model.pkl")
            setattr(self, f"{name}_model", model)
            self._loaded_versions[name] = model_dir.name
            logger.info("model_loaded", model=name, version=model_dir.name)
        except FileNotFoundError:
            logger.warning("model_not_found", model=name,
                           msg="Model not yet trained — endpoint will return error")
        except Exception as e:
            logger.error("model_load_error", model=name, error=str(e))

    @property
    def uptime_seconds(self) -> int:
        return int(time.time() - self._start_time)

    # ── Disruption Prediction ──

    def predict_disruption(self, features: dict) -> dict:
        """
        Run disruption prediction.
        Input: dict of feature values matching training feature order.
        Output: dict with probability, type, severity, factors.
        """
        if self.disruption_model is None:
            raise RuntimeError("Disruption model not loaded")

        # Build feature vector in the order the model expects
        feature_names = [
            "temperature_celsius", "rainfall_mm", "wind_speed_kmh", "humidity_pct",
            "congestion_index", "port_utilization_pct", "news_sentiment_score",
            "strike_mention_count_24h", "hour_of_day", "day_of_week",
            "is_dfc_corridor", "is_coastal_corridor",
            "corridor_avg_delay_hours_30d", "corridor_disruption_rate_90d",
            # One-hot encoded fields appended during feature engineering
        ]

        X = np.array([[features.get(f, 0.0) for f in feature_names]])

        # Probability prediction
        prob = float(self.disruption_model.predict_proba(X)[0, 1])

        # Severity mapping
        if prob < 0.3:
            severity = "LOW"
        elif prob < 0.6:
            severity = "MEDIUM"
        elif prob < 0.8:
            severity = "HIGH"
        else:
            severity = "CRITICAL"

        # Feature importance for explainability
        importances = self.disruption_model.feature_importances_
        top_indices = np.argsort(importances)[-3:][::-1]
        top_factors = [feature_names[i] for i in top_indices if i < len(feature_names)]

        return {
            "disruption_probability": prob,
            "predicted_disruption_type": self._infer_disruption_type(features, prob),
            "severity": severity,
            "contributing_factors": top_factors,
            "model_confidence": float(max(prob, 1 - prob)),
            "model_version": f"disruption/{self._loaded_versions.get('disruption', 'unknown')}",
        }

    def _infer_disruption_type(self, features: dict, prob: float) -> str:
        """Infer most likely disruption type from feature values."""
        if features.get("rainfall_mm", 0) > 50 or features.get("wind_speed_kmh", 0) > 60:
            return "weather_heavy_rain" if features.get("rainfall_mm", 0) > 50 else "weather_cyclone"
        if features.get("strike_mention_count_24h", 0) > 3:
            return "strike_transport"
        if features.get("congestion_index", 0) > 0.8:
            return "port_congestion"
        if features.get("news_sentiment_score", 0) < -0.5:
            return "policy_change"
        return "weather_heavy_rain"  # default

    # ── ETA Prediction ──

    def predict_eta(self, features: dict) -> dict:
        """
        Run ETA confidence band prediction.
        Returns p50, p75, p90, p99 delivery hours.
        """
        if self.eta_model is None:
            raise RuntimeError("ETA model not loaded")

        feature_names = [
            "total_distance_km", "cargo_weight_tonnes", "num_mode_transfers",
            "route_disruption_score", "carrier_ontime_rate_90d",
            "lane_avg_transit_hours",
            # Encoded features added during feature engineering
        ]

        X = np.array([[features.get(f, 0.0) for f in feature_names]])

        # Quantile predictions
        predictions = {}
        for quantile_name in ["p50", "p75", "p90", "p99"]:
            # QRF or similar quantile model
            q_val = float(self.eta_model.predict(X)[0])
            # Adjust based on quantile (simplified — real QRF returns all quantiles)
            predictions[quantile_name] = q_val

        return {
            **{f"{k}_hours": v for k, v in predictions.items()},
            "confidence_width_hours": predictions["p99"] - predictions["p50"],
            "model_version": f"eta/{self._loaded_versions.get('eta', 'unknown')}",
        }

    # ── Anomaly Detection ──

    def score_anomaly(self, features: dict) -> dict:
        """
        Score a shipment for anomalies.
        Uses Isolation Forest anomaly scores.
        """
        if self.anomaly_model is None:
            raise RuntimeError("Anomaly model not loaded")

        feature_names = [
            "route_deviation_km", "max_stationary_hours",
            "weight_delta_pct", "transit_time_ratio",
            "avg_speed_kmh", "stop_count",
        ]

        X = np.array([[features.get(f, 0.0) for f in feature_names]])

        # Isolation Forest: decision_function returns negative for anomalies
        raw_score = float(self.anomaly_model.decision_function(X)[0])
        # Normalise to 0–1 where 1 = highly anomalous
        anomaly_score = max(0.0, min(1.0, 0.5 - raw_score))

        anomaly_type = self._classify_anomaly(features, anomaly_score)
        should_alert = anomaly_score > config.ANOMALY_THRESHOLD

        return {
            "anomaly_score": anomaly_score,
            "anomaly_type": anomaly_type,
            "explanation": self._explain_anomaly(features, anomaly_type),
            "should_alert": should_alert,
            "severity": "HIGH" if anomaly_score > 0.85 else ("MEDIUM" if anomaly_score > 0.7 else "LOW"),
            "model_version": f"anomaly/{self._loaded_versions.get('anomaly', 'unknown')}",
        }

    def _classify_anomaly(self, features: dict, score: float) -> str:
        """Classify the type of anomaly based on feature deviations."""
        if score < config.ANOMALY_THRESHOLD:
            return "none"
        if features.get("weight_delta_pct", 0) > 10:
            return "weight_mismatch"
        if features.get("route_deviation_km", 0) > 50:
            return "route_deviation"
        if features.get("max_stationary_hours", 0) > 8:
            return "stationary_alert"
        if features.get("transit_time_ratio", 1.0) > 1.5:
            return "timing_anomaly"
        return "speed_anomaly"

    def _explain_anomaly(self, features: dict, anomaly_type: str) -> str:
        """Generate human-readable explanation for the anomaly."""
        explanations = {
            "none": "No anomaly detected. Shipment is within normal parameters.",
            "weight_mismatch": f"Weight discrepancy of {features.get('weight_delta_pct', 0):.1f}% detected at checkpoint.",
            "route_deviation": f"Shipment deviated {features.get('route_deviation_km', 0):.1f} km from planned route.",
            "stationary_alert": f"Shipment stationary for {features.get('max_stationary_hours', 0):.1f} hours at non-hub location.",
            "timing_anomaly": f"Transit time is {features.get('transit_time_ratio', 1.0):.1f}x the expected duration.",
            "speed_anomaly": f"Average speed of {features.get('avg_speed_kmh', 0):.1f} km/h is outside normal range.",
        }
        return explanations.get(anomaly_type, "Unknown anomaly type detected.")

    # ── Supplier Risk (Rule-based V1) ──

    @staticmethod
    def score_supplier_risk(inputs: list[dict]) -> list[dict]:
        """
        V1 rule-based supplier risk scoring.
        No trained model needed — weighted formula computed in Python.
        """
        results = []
        for inp in inputs:
            # Weighted risk components (weights sum to 1.0)
            weather_risk = (
                inp.get("location_flood_frequency", 0) * 10 +
                inp.get("location_cyclone_exposure", 0) * 30 +
                (20 if inp.get("is_coastal", False) else 0)
            ) * 0.20

            performance_risk = (
                (1 - inp.get("ontime_delivery_rate_90d", 0.9)) * 60 +
                min(inp.get("avg_delay_hours", 0) / 24, 1.0) * 40
            ) * 0.30

            financial_risk = (
                max(0, -inp.get("news_sentiment_30d", 0)) * 30 +
                (30 if not inp.get("gst_compliant", True) else 0) +
                min(inp.get("negative_news_count_30d", 0) / 5, 1.0) * 40
            ) * 0.25

            dependency_risk = (
                (100 - min(inp.get("num_ports_used", 1), 5) * 20) * 0.4 +
                inp.get("single_port_dependency_pct", 50) * 0.6
            ) * 0.25

            raw_score = weather_risk + performance_risk + financial_risk + dependency_risk
            score = max(0, min(100, raw_score))

            # Determine level
            level = "GREEN"
            for lev, (lo, hi) in RISK_THRESHOLDS.items():
                if lo <= score <= hi:
                    level = lev
                    break

            # Top risk factors
            factors_ranked = sorted([
                ("weather_exposure", weather_risk),
                ("delivery_performance", performance_risk),
                ("financial_signals", financial_risk),
                ("port_dependency", dependency_risk),
            ], key=lambda x: x[1], reverse=True)

            results.append({
                "supplier_id": inp.get("supplier_id", "unknown"),
                "risk_score": round(score, 1),
                "risk_level": level,
                "top_risk_factors": [f[0] for f in factors_ranked[:3]],
                "trend": "stable",  # V1 doesn't track trend
            })

        return results


# Singleton registry instance
registry = ModelRegistry()
```

### Task 2.4 — Write `ml/server/grpc_server.py`

```python
"""gRPC server implementation for NexusIQ ML inference service."""

import asyncio
from concurrent import futures

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_health.v1.health import HealthServicer

from server.config import config
from server.inference import registry
from utils.logger import logger

# These will be generated from proto — import after running grpc_tools
# python -m grpc_tools.protoc -I proto/ --python_out=server/ --grpc_python_out=server/ proto/inference.proto
try:
    import server.inference_pb2 as pb2
    import server.inference_pb2_grpc as pb2_grpc
except ImportError:
    logger.warning("grpc_stubs_not_found",
                   msg="Run 'make proto' to generate gRPC stubs from proto/inference.proto")
    pb2 = None
    pb2_grpc = None


class InferenceServicer(pb2_grpc.InferenceServiceServicer):
    """Implements the InferenceService gRPC interface."""

    def PredictDisruption(self, request, context):
        """Handle disruption prediction requests."""
        try:
            features = {
                "temperature_celsius": request.temperature_celsius,
                "rainfall_mm": request.rainfall_mm,
                "wind_speed_kmh": request.wind_speed_kmh,
                "humidity_pct": request.humidity_pct,
                "congestion_index": request.congestion_index,
                "port_utilization_pct": request.port_utilization_pct,
                "news_sentiment_score": request.news_sentiment_score,
                "strike_mention_count_24h": request.strike_mention_count_24h,
                "hour_of_day": request.hour_of_day,
                "day_of_week": request.day_of_week,
                "is_dfc_corridor": int(request.is_dfc_corridor),
                "is_coastal_corridor": int(request.is_coastal_corridor),
                "corridor_avg_delay_hours_30d": request.corridor_avg_delay_hours_30d,
                "corridor_disruption_rate_90d": request.corridor_disruption_rate_90d,
            }

            result = registry.predict_disruption(features)

            return pb2.DisruptionResponse(
                disruption_probability=result["disruption_probability"],
                predicted_disruption_type=result["predicted_disruption_type"],
                severity=result["severity"],
                contributing_factors=result["contributing_factors"],
                model_confidence=result["model_confidence"],
                model_version=result["model_version"],
            )
        except Exception as e:
            logger.error("disruption_prediction_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.DisruptionResponse()

    def PredictETA(self, request, context):
        """Handle ETA prediction requests."""
        try:
            features = {
                "total_distance_km": request.total_distance_km,
                "cargo_weight_tonnes": request.cargo_weight_tonnes,
                "num_mode_transfers": request.num_mode_transfers,
                "route_disruption_score": request.route_disruption_score,
                "carrier_ontime_rate_90d": request.carrier_ontime_rate_90d,
                "lane_avg_transit_hours": request.lane_avg_transit_hours,
            }

            result = registry.predict_eta(features)

            return pb2.ETAResponse(
                p50_hours=result["p50_hours"],
                p75_hours=result["p75_hours"],
                p90_hours=result["p90_hours"],
                p99_hours=result["p99_hours"],
                confidence_width_hours=result["confidence_width_hours"],
                model_version=result["model_version"],
            )
        except Exception as e:
            logger.error("eta_prediction_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.ETAResponse()

    def ScoreAnomaly(self, request, context):
        """Handle anomaly scoring requests."""
        try:
            # Compute derived features from GPS track
            gps_points = list(request.gps_track)
            route_deviation = request.actual_distance_km - request.planned_distance_km
            weight_delta = (
                abs(request.measured_weight_kg - request.declared_weight_kg)
                / max(request.declared_weight_kg, 1) * 100
            )
            transit_ratio = (
                request.actual_elapsed_hours / max(request.expected_transit_hours, 1)
            )

            # Find max stationary time from GPS
            max_stationary = 0.0
            if gps_points:
                speeds = [p.speed_kmh for p in gps_points]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0
            else:
                avg_speed = 0.0

            features = {
                "route_deviation_km": max(0, route_deviation),
                "max_stationary_hours": max_stationary,
                "weight_delta_pct": weight_delta,
                "transit_time_ratio": transit_ratio,
                "avg_speed_kmh": avg_speed,
                "stop_count": sum(1 for p in gps_points if p.is_stopped),
            }

            result = registry.score_anomaly(features)

            return pb2.AnomalyResponse(
                anomaly_score=result["anomaly_score"],
                anomaly_type=result["anomaly_type"],
                explanation=result["explanation"],
                should_alert=result["should_alert"],
                severity=result["severity"],
                model_version=result["model_version"],
            )
        except Exception as e:
            logger.error("anomaly_scoring_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.AnomalyResponse()

    def BatchScoreRisk(self, request, context):
        """Handle batch supplier risk scoring."""
        try:
            inputs = []
            for s in request.suppliers:
                inputs.append({
                    "supplier_id": s.supplier_id,
                    "location_flood_frequency": s.location_flood_frequency,
                    "location_cyclone_exposure": s.location_cyclone_exposure,
                    "is_coastal": s.is_coastal,
                    "ontime_delivery_rate_90d": s.ontime_delivery_rate_90d,
                    "avg_delay_hours": s.avg_delay_hours,
                    "total_shipments_90d": s.total_shipments_90d,
                    "news_sentiment_30d": s.news_sentiment_30d,
                    "gst_compliant": s.gst_compliant,
                    "negative_news_count_30d": s.negative_news_count_30d,
                    "num_ports_used": s.num_ports_used,
                    "single_port_dependency_pct": s.single_port_dependency_pct,
                })

            results = registry.score_supplier_risk(inputs)

            response = pb2.RiskBatchResponse(scorer_version="risk/v1")
            for r in results:
                response.results.append(pb2.SupplierRiskOutput(
                    supplier_id=r["supplier_id"],
                    risk_score=r["risk_score"],
                    risk_level=r["risk_level"],
                    top_risk_factors=r["top_risk_factors"],
                    trend=r["trend"],
                ))
            return response
        except Exception as e:
            logger.error("risk_scoring_error", error=str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return pb2.RiskBatchResponse()

    def HealthCheck(self, request, context):
        """Return model loading status and server health."""
        return pb2.HealthResponse(
            healthy=True,
            disruption_model_loaded=registry.disruption_model is not None,
            eta_model_loaded=registry.eta_model is not None,
            anomaly_model_loaded=registry.anomaly_model is not None,
            risk_scorer_ready=True,  # Rule-based, always ready
            server_version="0.1.0",
            uptime_seconds=registry.uptime_seconds,
        )


def serve(port: int | None = None) -> grpc.Server:
    """Start the gRPC server."""
    if pb2_grpc is None:
        raise RuntimeError("gRPC stubs not generated. Run 'make proto' first.")

    port = port or config.GRPC_PORT
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Register inference service
    pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)

    # Register standard gRPC health check
    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("nexusiq.ml.InferenceService", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("grpc_server_started", port=port)
    return server
```

### Task 2.5 — Write `ml/server/health.py`

```python
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
```

### Task 2.6 — Write `ml/server/app.py`

```python
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
```

### Task 2.7 — Write `ml/pipelines/base.py`

This is the abstract base class that all training pipelines inherit from. It enforces the consistent pattern: load_data → engineer_features → split → train → evaluate → save.

```python
"""
Base pipeline class — enforces consistent ML workflow across all models.
Every pipeline (disruption, eta, anomaly) inherits from this and implements the abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from utils.io import save_model, save_parquet, load_parquet, FEATURES_DIR, MODELS_DIR
from utils.metrics import save_eval_report
from utils.logger import logger
from utils.constants import RANDOM_SEED


class BasePipeline(ABC):
    """Abstract base class for all NexusIQ ML training pipelines."""

    def __init__(self, model_name: str, version: int = 1):
        self.model_name = model_name
        self.version = version
        self.model: Any = None
        self.metrics: dict = {}
        self.feature_names: list[str] = []

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load raw or synthetic data for this pipeline."""
        ...

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Transform raw data into feature matrix.
        Returns: (feature_dataframe, list_of_feature_column_names)
        """
        ...

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """Train the model. Return the trained model object."""
        ...

    @abstractmethod
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on test set. Return metrics dict."""
        ...

    @abstractmethod
    def get_target_column(self) -> str:
        """Return the name of the target column in the feature dataframe."""
        ...

    def split_data(self, df: pd.DataFrame, feature_cols: list[str],
                   test_size: float = 0.15, val_size: float = 0.15,
                   time_column: str | None = None) -> tuple:
        """
        Split data into train/val/test sets.
        If time_column is provided, uses time-based split (no future leakage).
        Otherwise, stratified random split.
        """
        target_col = self.get_target_column()
        X = df[feature_cols].values
        y = df[target_col].values

        if time_column and time_column in df.columns:
            # Time-based split
            df_sorted = df.sort_values(time_column)
            n = len(df_sorted)
            train_end = int(n * (1 - test_size - val_size))
            val_end = int(n * (1 - test_size))

            X_train = df_sorted[feature_cols].values[:train_end]
            y_train = df_sorted[target_col].values[:train_end]
            X_val = df_sorted[feature_cols].values[train_end:val_end]
            y_val = df_sorted[target_col].values[train_end:val_end]
            X_test = df_sorted[feature_cols].values[val_end:]
            y_test = df_sorted[target_col].values[val_end:]
        else:
            # Stratified random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
            )
            relative_val = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=relative_val, random_state=RANDOM_SEED, stratify=y_temp
            )

        logger.info("data_split",
                     model=self.model_name,
                     train=len(X_train), val=len(X_val), test=len(X_test))
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run(self) -> dict:
        """
        Execute the full pipeline: load → features → split → train → evaluate → save.
        Returns the evaluation metrics.
        """
        logger.info("pipeline_start", model=self.model_name, version=self.version)

        # 1. Load data
        df = self.load_data()
        logger.info("data_loaded", rows=len(df), columns=len(df.columns))

        # 2. Engineer features
        df_features, feature_cols = self.engineer_features(df)
        self.feature_names = feature_cols
        logger.info("features_engineered", feature_count=len(feature_cols))

        # Save feature matrix
        features_path = FEATURES_DIR / f"{self.model_name}_features.parquet"
        save_parquet(df_features, features_path)

        # 3. Split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            df_features, feature_cols
        )

        # 4. Train
        self.model = self.train(X_train, y_train, X_val, y_val)
        logger.info("model_trained", model=self.model_name)

        # 5. Evaluate
        self.metrics = self.evaluate(self.model, X_test, y_test)
        logger.info("model_evaluated", model=self.model_name, metrics=self.metrics)

        # 6. Save
        save_model(
            model=self.model,
            model_name=self.model_name,
            version=self.version,
            metrics=self.metrics,
            hyperparams=getattr(self, "best_params", {}),
            dataset_path=features_path,
        )
        save_eval_report(self.metrics, self.model_name, self.version)

        logger.info("pipeline_complete", model=self.model_name, version=self.version)
        return self.metrics
```

### Task 2.8 — Write `ml/Makefile`

```makefile
.PHONY: setup proto test lint clean train-all serve docker-build docker-run

# ── Environment ──
PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

setup: ## Create venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "✅ Environment ready. Activate with: source $(VENV)/bin/activate"

# ── gRPC ──
proto: ## Generate Python gRPC stubs from proto file
	$(PY) -m grpc_tools.protoc \
		-I proto/ \
		--python_out=server/ \
		--grpc_python_out=server/ \
		proto/inference.proto
	@echo "✅ gRPC stubs generated in server/"

# ── Data ──
data-download: ## Download real data sources (IMD, OpenWeather, etc.)
	$(PY) -m scripts.download_imd_weather
	$(PY) -m scripts.download_openweather
	$(PY) -m scripts.download_port_data
	$(PY) -m scripts.download_news

data-synthetic: ## Generate all synthetic datasets
	$(PY) -m scripts.generate_shipments
	$(PY) -m scripts.generate_gps_telemetry
	$(PY) -m scripts.generate_port_events
	$(PY) -m scripts.generate_weather_disruptions
	$(PY) -m scripts.generate_suppliers

data-all: data-download data-synthetic ## Download + generate all data

# ── Training ──
train-disruption: ## Train disruption prediction model
	$(PY) -m pipelines.disruption.train

train-eta: ## Train ETA confidence band model
	$(PY) -m pipelines.eta.train

train-anomaly: ## Train anomaly detection model
	$(PY) -m pipelines.anomaly.train

train-all: train-disruption train-eta train-anomaly ## Train all models

# ── Server ──
serve: ## Start ML sidecar (gRPC + HTTP health)
	$(PY) -m server.app

serve-dev: ## Start with auto-reload (dev only)
	$(PY) -m uvicorn server.app:create_app --factory --reload --port 8081

# ── Testing ──
test: ## Run all tests
	$(PY) -m pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PY) -m pytest tests/ -v --cov=. --cov-report=term-missing

# ── Quality ──
lint: ## Run linter
	$(PY) -m ruff check .

lint-fix: ## Auto-fix lint errors
	$(PY) -m ruff check . --fix

typecheck: ## Run type checker
	$(PY) -m mypy server/ pipelines/ utils/

# ── Docker ──
docker-build: ## Build ML sidecar Docker image
	docker build -t nexusiq-ml-sidecar:latest .

docker-run: ## Run ML sidecar container
	docker run -p 50051:50051 -p 8081:8081 \
		-v $(PWD)/models:/app/models \
		nexusiq-ml-sidecar:latest

# ── Cleanup ──
clean: ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -f server/inference_pb2.py server/inference_pb2_grpc.py

clean-models: ## Remove all trained model artifacts
	rm -rf models/disruption/v* models/eta/v* models/anomaly/v*

clean-data: ## Remove all downloaded and generated data
	rm -rf data/raw/* data/clean/* data/features/* data/synthetic/*
	touch data/raw/.gitkeep data/clean/.gitkeep data/features/.gitkeep data/synthetic/.gitkeep

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
```

### Task 2.9 — Write `ml/Dockerfile`

```dockerfile
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
```

### Task 2.10 — Write `ml/docker-compose.ml.yaml`

This is a standalone compose file for ML development — lets you spin up the sidecar + dependencies without the full NexusIQ stack.

```yaml
version: "3.8"

services:
  ml-sidecar:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "50051:50051"   # gRPC
      - "8081:8081"     # HTTP health
    volumes:
      - ./models:/app/models        # persist trained models
      - ./data:/app/data            # access training data
    environment:
      - GRPC_PORT=50051
      - HTTP_PORT=8081
      - MODEL_DIR=/app/models
      - LOG_LEVEL=INFO
      - DISRUPTION_THRESHOLD=0.6
      - ANOMALY_THRESHOLD=0.7
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; r=requests.get('http://localhost:8081/health'); exit(0 if r.ok else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
```

### Task 2.11 — Write `ml/README.md`

```markdown
# NexusIQ ML Pipeline

Machine learning models and inference sidecar for the NexusIQ supply chain intelligence platform.

## Quick Start

```bash
# 1. Set up Python environment
make setup
source .venv/bin/activate

# 2. Generate gRPC stubs
make proto

# 3. Generate synthetic training data
make data-synthetic

# 4. Train all models
make train-all

# 5. Start the inference server
make serve
```

## Architecture

The ML sidecar runs as a separate container, exposing gRPC endpoints consumed by Go backend services.

| Endpoint | Model | Algorithm | Latency Target |
|---|---|---|---|
| PredictDisruption | Disruption Predictor | XGBoost | < 50ms |
| PredictETA | ETA Confidence Bands | Quantile Regression Forest | < 80ms |
| ScoreAnomaly | Anomaly Detector | Isolation Forest | < 100ms |
| BatchScoreRisk | Supplier Risk | Rule-based v1 | < 500ms (batch 50) |

## Commands

Run `make help` for all available commands.

## Testing

```bash
make test        # run all tests
make test-cov    # with coverage report
make lint        # check code style
```
```

---

## VERIFICATION CHECKLIST

After completing all tasks, verify:

1. **Directory structure** — run `find ml/ -type f | head -60` and confirm all files exist
2. **Python env** — run `cd ml && make setup && source .venv/bin/activate`
3. **gRPC stubs** — run `make proto` and confirm `server/inference_pb2.py` and `server/inference_pb2_grpc.py` are generated
4. **Imports work** — run `python -c "from utils.constants import LOGISTICS_HUBS; print(f'{len(LOGISTICS_HUBS)} hubs loaded')"`
5. **Server starts** — run `make serve` (will warn about missing models but should not crash)
6. **Health endpoint** — `curl http://localhost:8081/health` returns `{"status": "ok"}`
7. **Readiness endpoint** — `curl http://localhost:8081/ready` returns model status (all false until trained)
8. **Lint passes** — `make lint` shows no errors
9. **Docker builds** — `make docker-build` completes successfully

If all 9 checks pass, Block 1 & 2 are complete. You're ready for Block 3 (Data Ingestion & Synthetic Data Generation).

---

## IMPORTANT NOTES FOR CLAUDE CODE

- Do NOT add tensorflow, torch, or any deep learning dependencies at this stage
- Do NOT create any Jupyter notebooks — all code is in `.py` files
- Do NOT use relative imports — use absolute imports from the `ml/` root (e.g., `from utils.logger import logger`)
- Make sure every `__init__.py` file exists (empty is fine)
- The `proto/inference.proto` file is the SINGLE SOURCE OF TRUTH for the API contract. Both Go and Python generate code from it.
- All config values come from environment variables via `server/config.py` — no hardcoded values in server code
- Feature names in `server/inference.py` MUST match what the training pipeline produces — this is enforced when we build Block 3
