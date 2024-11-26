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
    # Find the center point of each crosswalk
    #   - Turns complex crosswalk shapes into simple points
    #   - Gets (x,y) location for each crosswalk
    crosswalk_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                           for geometry in crosswalks_df['geometry']]

    # Same thing for lanes - find center of each road lane
    lane_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                      for geometry in lanes_df['geometry']]

    # Same thing for walkways - find center of each sidewalk
    walkway_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                         for geometry in walkways_df['geometry']]

    # Put all the points together in one list
    features = crosswalk_positions + lane_positions + walkway_positions
    # Turn the list of points into a tensor
    environment_features = torch.tensor(features, dtype=torch.float32)

    print("Original environment features shape:", environment_features.shape)

    # If we have too many features, just keep the first 100
    #   - 100 features is usually enough to understand the scene
    # if environment_features.shape[0] > 5000:
    #     environment_features = environment_features[:5000]

    # Add a batch dimension because our model expects it
    environment_tensor = environment_features.unsqueeze(0).to(torch.float32)

    print("Final environment tensor shape:", environment_tensor.shape)
    return environment_tensor
