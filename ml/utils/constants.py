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
