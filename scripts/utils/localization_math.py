import math
import numpy as np

MEAN_EARTH_RADIUS = 6371

# haversine equation for circle distance in km, chapter 3.6 page 50
def haversine_km(lon, lat, rlon, rlat) -> np.ndarray:
    rlon, rlat = np.radians(rlon), np.radians(rlat)
    lon, lat = math.radians(lon), math.radians(lat)
    haversine_radicand = np.sin((rlat - lat) / 2) ** 2 + math.cos(lat) * np.cos(rlat) * np.sin((rlon - lon) / 2) ** 2
    haversine_distance = 2 * MEAN_EARTH_RADIUS * np.arcsin(np.sqrt(haversine_radicand))
    return haversine_distance


# haversine distance in km to the nearest reference image, chapter 3.6
def nearest_reference_km(lon, lat, ref_coords) -> float:
    return float(haversine_km(lon, lat, ref_coords[:, 0], ref_coords[:, 1]).min())
