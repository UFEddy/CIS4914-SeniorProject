import os
import pandas as pd
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper

# paths and configuration variables
NUPLAN_DATA_ROOT = "/Users/eddyrosales/Documents/Senior Project/NuPlan/nuplan-devkit/nuplan/dataset/"
NUPLAN_MAPS_ROOT = "/Users/eddyrosales/Documents/Senior Project/NuPlan/nuplan-devkit/nuplan/dataset/maps"
NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"

# Initialize NuPlanDBWrapper
nuplandb_wrapper = NuPlanDBWrapper(
    data_root=NUPLAN_DATA_ROOT,
    map_root=NUPLAN_MAPS_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_version=NUPLAN_MAP_VERSION,
)
map_api = nuplandb_wrapper.maps_db

OUTPUT_DIR = "map_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_map_layer(layer_name: str):
    """Extracts a specific map layer and saves to CSV."""
    try:
        layer_data = map_api.load_vector_layer("sg-one-north", layer_name)
        output_path = os.path.join(OUTPUT_DIR, f"{layer_name}.csv")
        layer_data.to_csv(output_path, index=False)
        print(f"Extracted {layer_name} to {output_path}")
    except Exception as e:
        print(f"Failed to extract {layer_name}: {e}")

layers = [
    "generic_drivable_areas",  # Drivable areas
    "road_segments",           # Road block segments
    "walkways",                # Pedestrian walkways
    "crosswalks",              # Pedestrian crossings
    "lanes_polygons",          # Lanes
    "lane_connectors"          # Lane connectors
]

for layer in layers:
    extract_map_layer(layer)
