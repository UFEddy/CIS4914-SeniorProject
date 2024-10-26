import os
import csv
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper

# paths and configuration variables
NUPLAN_DATA_ROOT = "PATH FOR Dataset"
NUPLAN_MAPS_ROOT = "PATH FOR Maps"
NUPLAN_DB_FILES = f"{NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini"
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"

# Initialize NuPlanDBWrapper
nuplandb_wrapper = NuPlanDBWrapper(
    data_root=NUPLAN_DATA_ROOT,
    map_root=NUPLAN_MAPS_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_version=NUPLAN_MAP_VERSION,
)

# Output directory for CSV files
OUTPUT_DIR = "./log_data_csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_table_to_csv(table, filename, fields):
    """Extracts data from a given table and writes it to a CSV."""
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in table:
            writer.writerow({field: getattr(record, field, None) for field in fields})
    print(f"Extracted {filename} to {output_path}")

# Load a specific log database
log_name = "2021.05.12.22.00.38_veh-35_01008_01518" 
log_db = nuplandb_wrapper.get_log_db(log_name)

# Extract data from log tables
extract_table_to_csv(log_db.lidar_box, "lidar_box.csv", [
    "token", "lidar_pc_token", "track_token", "x", "y", "z", "vx", "vy", "vz", 
    "yaw", "width", "length", "height", "confidence"
])

extract_table_to_csv(log_db.track, "track.csv", ["token", "width", "length", "height"])

extract_table_to_csv(log_db.ego_pose, "ego_pose.csv", [
    "timestamp", "x", "y", "z", "vx", "vy", "vz", 
    "acceleration_x", "acceleration_y", "acceleration_z", 
    "angular_rate_x", "angular_rate_y", "angular_rate_z", 
    "qw", "qx", "qy", "qz"
])

extract_table_to_csv(log_db.lidar_pc, "lidar_pc.csv", [
    "token", "ego_pose_token", "timestamp"
])

extract_table_to_csv(log_db.scenario_tag, "scenario_tag.csv", [
    "token", "lidar_pc_token", "type", "agent_track_token"
])

extract_table_to_csv(log_db.traffic_light_status, "traffic_light_status.csv", [
    "token", "lidar_pc_token", "lane_connector_id", "status"
])

extract_table_to_csv(log_db.scene, "scene.csv", [
    "token", "goal_ego_pose_token", "roadblock_ids"
])

extract_table_to_csv(log_db.category, "category.csv", ["token", "name"])

extract_table_to_csv(log_db.lidar, "lidar.csv", [
    "token", "translation", "rotation", "model"
])

extract_table_to_csv(log_db.camera, "camera.csv", [
    "token", "translation", "rotation", "intrinsic", "distortion", 
    "width", "height"
])

print("Log data extraction completed.")
