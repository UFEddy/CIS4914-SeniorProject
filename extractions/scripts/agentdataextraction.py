import os
import csv
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper

# paths and configuration variables
NUPLAN_DATA_ROOT = "PATH to DATASET ROOT"
NUPLAN_MAPS_ROOT = "PATH TO DATASET MAP ROOT"
NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"


# Initialize NuPlanDBWrapper
nuplandb_wrapper = NuPlanDBWrapper(
    data_root=NUPLAN_DATA_ROOT,
    map_root=NUPLAN_MAPS_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_version=NUPLAN_MAP_VERSION,
)

OUTPUT_DIR = "agent_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_agent_data(log_db):
    """Extracts track and lidar_box data to CSV."""
    output_path = os.path.join(OUTPUT_DIR, "agent_data.csv")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "track_token", "category", "x", "y", "z", "vx", "vy", "vz", "yaw", 
            "width", "length", "height", "confidence"
        ])
        writer.writeheader()

        # Loop over lidar_boxes and get track
        for box in log_db.lidar_box:
            track = log_db.track[box.track_token]
            category_name = log_db.category[track.category_token].name

            writer.writerow({
                "track_token": box.track_token,
                "category": category_name,
                "x": box.x, "y": box.y, "z": box.z,
                "vx": box.vx, "vy": box.vy, "vz": box.vz,
                "yaw": box.yaw,
                "width": box.width, "length": box.length, "height": box.height,
                "confidence": box.confidence
            })

    print(f"Agent data saved to {output_path}")

# Load log database
log_db_name = "2021.05.12.22.00.38_veh-35_01008_01518"
log_db = nuplandb_wrapper.get_log_db(log_db_name)

extract_agent_data(log_db)
