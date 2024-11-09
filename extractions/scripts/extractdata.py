import os
import csv
from nuplan.database.nuplan_db_orm.nuplandb_wrapper import NuPlanDBWrapper
from dotenv import load_dotenv

load_dotenv()

# paths and configuration variables
NUPLAN_DATA_ROOT = os.getenv("NUPLAN_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.getenv("NUPLAN_MAPS_ROOT")
NUPLAN_DB_FILES = os.getenv("NUPLAN_DB_FILES")
NUPLAN_MAP_VERSION = os.getenv("NUPLAN_MAP_VERSION")
LOGNAME = "2021.05.12.22.00.38_veh-35_01008_01518" 

# Initialize NuPlanDBWrapper
nuplandb_wrapper = NuPlanDBWrapper(
    data_root=NUPLAN_DATA_ROOT,
    map_root=NUPLAN_MAPS_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_version=NUPLAN_MAP_VERSION,
)

# Output directory for CSV files
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_table_to_csv(table, filename, fields):
    """Extracts data from table and write CSV."""
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in table:
            writer.writerow({field: getattr(record, field, None) for field in fields})
    print(f"Extracted {filename} to {output_path}")

# Load a specific log database
log_name = LOGNAME
log_db = nuplandb_wrapper.get_log_db(log_name)

# Extract data from log tables #

# # Extract log info
# extract_table_to_csv(log_db.log, "log.csv", [
#     "token", "vehicle_name", "date",
#     "timestamp", "logfile", "location", "map_version"
# ])

# Extract ego_pose
extract_table_to_csv(log_db.ego_pose, "ego_pose.csv", [
    "token", "log_token",
    "timestamp", 
    f"x", "y", "z", 
    "qw", "qx", "qy", "qz",
    "vx", "vy", "vz", 
    "acceleration_x", "acceleration_y", "acceleration_z", 
    "angular_rate_x", "angular_rate_y", "angular_rate_z", 
    "epsg"
])

# Extract camera
extract_table_to_csv(log_db.camera, "camera.csv", [
    "token", "log_token", "channel", "model", "translation", "rotation", "intrinsic", "distortion", 
    "width", "height"
])

# Extract image
extract_table_to_csv(log_db.image, "image.csv", [
    "token", "next_token", "prev_token",
    "ego_pose_token", "camera_token", "filename_jpg", "timestamp"
])

# Extract lidar
extract_table_to_csv(log_db.lidar, "lidar.csv", [
    "token", "log_token", "channel", "model", "translation", "rotation"
])

#Extract lidar_pc
extract_table_to_csv(log_db.lidar_pc, "lidar_pc.csv", [
    "token", "next_token", "prev_token", "scene_token", "ego_pose_token",
    "lidar_token", "filename", "timestamp"
])


#Extract libar_box
extract_table_to_csv(log_db.lidar_box, "lidar_box.csv", [
    "token", "lidar_pc_token", "track_token", "next_token", "prev_token", 
    "x", "y", "z", 
    "width", "length", "height",
    "vx", "vy", "vz", 
    "yaw",  "confidence"
])

#Extract track
extract_table_to_csv(log_db.track, "track.csv", [
    "token", "category_token", "width", "length", "height"
])

#extract category
extract_table_to_csv(log_db.category, "category.csv", [
    "token", "name", "description"
])

#Extract scene
extract_table_to_csv(log_db.scene, "scene.csv", [
    "token", "log_token", "name",
    "goal_ego_pose_token", "roadblock_ids"
])

#Extract scenario tag
extract_table_to_csv(log_db.scenario_tag, "scenario_tag.csv", [
    "token", "lidar_pc_token", "type", "agent_track_token"
])

# Extracdt traffic_light_status
extract_table_to_csv(log_db.traffic_light_status, "traffic_light_status.csv", [
    "token", "lidar_pc_token",
    "lane_connector_id", "status"
])

print("Log data extraction completed.")
