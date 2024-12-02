import torch
from shapely.wkt import loads
import numpy as np


def normalize_positions(positions, reference_position):
    """
    Safely normalize positions relative to a reference point
    """
    # Use a fixed scale rather than std to avoid instability
    POSITION_SCALE = 100.0  # 100 meters as characteristic length

    # Center around reference position
    normalized = (positions - reference_position) / POSITION_SCALE
    return normalized


def process_environment_data(crosswalks_df, lanes_df, walkways_df, ego_position):
    """
    Process environment DataFrames into tensor format for the encoder with stable normalization.
    """
    # Get reference position (mean of batch)
    reference_position = ego_position.mean(dim=0).cpu().numpy()

    # Fixed scales for normalization
    WIDTH_SCALE = 5.0  # meters
    SPEED_SCALE = 30.0  # m/s (~70mph)

    # Process lanes
    lane_features_list = []
    for _, lane in lanes_df.iterrows():
        try:
            polygon = loads(lane['geometry'])
            centroid = polygon.centroid

            # Calculate lane direction safely
            coords = list(polygon.exterior.coords)
            direction = np.array(coords[-1]) - np.array(coords[0])
            norm = np.linalg.norm(direction)
            if norm > 1e-6:  # Only normalize if magnitude is non-zero
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0])  # Default direction

            # Normalize position relative to ego
            pos = np.array([centroid.x, centroid.y])
            normalized_pos = (pos - reference_position) / 100.0  # Scale by 100m

            features = [
                normalized_pos[0],
                normalized_pos[1],
                direction[0],
                direction[1],
                np.clip(lane.get('width', 3.5) / WIDTH_SCALE, 0.1, 2.0),
                np.clip(lane.get('speed_limit', 13.41) / SPEED_SCALE, 0.1, 1.0),
                float(lane.get('intersection', False))
            ]
            lane_features_list.append(features)
        except (ValueError, AttributeError) as e:
            continue

    # Process crosswalks
    crosswalk_features_list = []
    for _, crosswalk in crosswalks_df.iterrows():
        try:
            polygon = loads(crosswalk['geometry'])
            centroid = polygon.centroid
            bounds = polygon.bounds

            # Normalize position
            pos = np.array([centroid.x, centroid.y])
            normalized_pos = (pos - reference_position) / 100.0

            features = [
                normalized_pos[0],
                normalized_pos[1],
                np.clip((bounds[2] - bounds[0]) / WIDTH_SCALE, 0.1, 2.0),
                np.clip((bounds[3] - bounds[1]) / WIDTH_SCALE, 0.1, 2.0),
                1.0
            ]
            crosswalk_features_list.append(features)
        except (ValueError, AttributeError) as e:
            continue

    # Process walkways
    walkway_features_list = []
    for _, walkway in walkways_df.iterrows():
        try:
            polygon = loads(walkway['geometry'])
            centroid = polygon.centroid
            bounds = polygon.bounds

            # Normalize position
            pos = np.array([centroid.x, centroid.y])
            normalized_pos = (pos - reference_position) / 100.0

            features = [
                normalized_pos[0],
                normalized_pos[1],
                np.clip((bounds[2] - bounds[0]) / WIDTH_SCALE, 0.1, 2.0),
                float(walkway.get('type', 'sidewalk') == 'sidewalk')
            ]
            walkway_features_list.append(features)
        except (ValueError, AttributeError) as e:
            continue

    # Convert to tensors with safety checks
    def safe_tensor_convert(features_list, feature_size):
        if not features_list:
            return torch.zeros((1, feature_size), dtype=torch.float32)
        tensor = torch.tensor(features_list, dtype=torch.float32)
        # Add dummy batch dimension if empty
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    lane_features = safe_tensor_convert(lane_features_list, 7)
    crosswalk_features = safe_tensor_convert(crosswalk_features_list, 5)
    walkway_features = safe_tensor_convert(walkway_features_list, 4)

    return lane_features, crosswalk_features, walkway_features


def prepare_environment_data(crosswalks_df, lanes_df, walkways_df):
    """
    Add any missing columns with default values
    """
    # Add defaults to lanes
    if 'width' not in lanes_df.columns:
        lanes_df['width'] = 12
    if 'speed_limit' not in lanes_df.columns:
        lanes_df['speed_limit'] = 25.0
    if 'intersection' not in lanes_df.columns:
        lanes_df['intersection'] = False

    # Add defaults to walkways
    if 'type' not in walkways_df.columns:
        walkways_df['type'] = 'sidewalk'

    return lanes_df, crosswalks_df, walkways_df
