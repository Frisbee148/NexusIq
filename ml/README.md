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
