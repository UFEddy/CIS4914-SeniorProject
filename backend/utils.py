import torch
from shapely.wkt import loads


# Encoding function
def encode_environment(crosswalks_df, lanes_df, walkways_df):
    """
    Encodes environmental features in a scene-centered frame (absolute global coordinates).

    Args:
        crosswalks_df (DataFrame): DataFrame with crosswalk polygons in global coordinates.
        lanes_df (DataFrame): DataFrame with lane polygons in global coordinates.
        walkways_df (DataFrame): DataFrame with walkway polygons in global coordinates.

    Returns:
        torch.Tensor: 3D tensor with shape (num_sequences, sequence_length, num_features).
    """
    # Convert WKT polygons to Shapely Polygons and calculate centroids
    crosswalk_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                           for geometry in crosswalks_df['geometry']]
    lane_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                      for geometry in lanes_df['geometry']]
    walkway_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                         for geometry in walkways_df['geometry']]

    # Combine all feature positions
    features = crosswalk_positions + lane_positions + walkway_positions
    environment_features = torch.tensor(features, dtype=torch.float32)

    print("Original environment features shape:", environment_features.shape)

    # Take a subset (first 100) of features if too large
    if environment_features.shape[0] > 100:
        environment_features = environment_features[:100]

    # Reshape to (batch_size=1, sequence_length, num_features)
    environment_tensor = environment_features.unsqueeze(0)

    print("Final environment tensor shape:", environment_tensor.shape)
    return environment_tensor
