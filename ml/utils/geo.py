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
