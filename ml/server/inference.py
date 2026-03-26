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
