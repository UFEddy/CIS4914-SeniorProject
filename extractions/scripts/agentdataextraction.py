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
NUPLAN_MAP_VERSION = "nuplan-maps-v1.0"
LOGNAME = "2021.05.12.22.00.38_veh-35_01008_01518" 


# Initialize NuPlanDBWrapper
nuplandb_wrapper = NuPlanDBWrapper(
    data_root=NUPLAN_DATA_ROOT,
    map_root=NUPLAN_MAPS_ROOT,
    db_files=NUPLAN_DB_FILES,
    map_version=NUPLAN_MAP_VERSION,
)

OUTPUT_DIR = "agent_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_agent_data(log_db, duration_seconds=120):
    """Extracts track and lidar_box data to CSV."""
    output_path = os.path.join(OUTPUT_DIR, "agent_data.csv")
    
    # Retrieve all lidar_boxes from the log
    lidar_boxes = log_db.lidar_box
    
    if not lidar_boxes:
        print("No lidar boxes found in the log.")
        return
    
    # Retrieve all corresponding lidar_pc entries
    lidar_pc_table = log_db.lidar_pc

    # Find the earliest timestamp from all lidar_pcs referenced by lidar_boxes
    cutoff_timestamp = None
    if duration_seconds is not None:
        earliest_timestamp = float('inf')
        for box in lidar_boxes:
            pc = lidar_pc_table[box.lidar_pc_token]
            if pc.timestamp < earliest_timestamp:
                earliest_timestamp = pc.timestamp

        # Calculate the cutoff timestamp (first 10 seconds)
        # The timestamp is in microseconds, so convert duration_seconds to microseconds
        cutoff_timestamp = earliest_timestamp + duration_seconds * 1e6

    with open(output_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            "timestamp", "track_token", "category", "x", "y", "z", "vx", "vy", "vz", "yaw", 
            "width", "length", "height", "confidence"
        ])
        writer.writeheader()

        # Loop over lidar_boxes and get track
        for box in log_db.lidar_box:
            # Find the corresponding lidar_pc to get the timestamp
            pc = lidar_pc_table[box.lidar_pc_token]
            current_timestamp = pc.timestamp

            # Check if the current box's timestamp is within the first `duration_seconds` seconds
            if cutoff_timestamp is not None and current_timestamp > cutoff_timestamp:
                continue  # Skip any box beyond the first `duration_seconds` seconds

            # Retrieve the track information
            track = log_db.track[box.track_token]
            category = log_db.category[track.category_token]


            writer.writerow({
                "timestamp": current_timestamp,
                "track_token": box.track_token,
                "category": category.name,
                "x": box.x,
                "y": box.y,
                "z": box.z,
                "vx": box.vx,
                "vy": box.vy,
                "vz": box.vz,
                "yaw": box.yaw,
                "width": box.width,
                "length": box.length,
                "height": box.height,
                "confidence": box.confidence
            })
    if duration_seconds is not None:
        print(f"Agent data for the first {duration_seconds} seconds saved to {output_path}")
    else:
        print(f"Agent data for the entire log saved to {output_path}")

# Load log database
log_db_name = LOGNAME
log_db = nuplandb_wrapper.get_log_db(log_db_name)

extract_agent_data(log_db, duration_seconds=120)
