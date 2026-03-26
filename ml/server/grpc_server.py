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
