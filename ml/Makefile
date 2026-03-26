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
