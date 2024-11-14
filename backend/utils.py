import pandas as pd
import torch
from shapely.wkt import loads
import matplotlib.pyplot as plt


def load_walkway_polygons(file_path):
    """
    Load walkway polygons from a CSV file.

    Args:
        file_path (str): Path to the walkways CSV file.

    Returns:
        list: A list of Shapely Polygon objects representing walkways.
    """
    walkways_df = pd.read_csv(file_path)
    walkways_df['geometry'] = walkways_df['geometry'].apply(loads)
    return walkways_df['geometry'].tolist()


# Encoding function
def encode_environment(crosswalks_df, lanes_df, walkways_df, sequence_length):
    """
    Encodes environmental features in a scene-centered frame (absolute global coordinates).

    Args:
        crosswalks_df (DataFrame): DataFrame with crosswalk polygons in global coordinates.
        lanes_df (DataFrame): DataFrame with lane polygons in global coordinates.
        walkways_df (DataFrame): DataFrame with walkway polygons in global coordinates.
        sequence_length (int): Number of time steps in each sequence.

    Returns:
        torch.Tensor: 3D tensor with shape (num_sequences, sequence_length, num_features).
    """
    environment_features = []

    # Convert WKT polygons to Shapely Polygons and calculate centroids for each feature
    crosswalk_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                           for geometry in crosswalks_df['geometry']]
    lane_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                      for geometry in lanes_df['geometry']]
    walkway_positions = [(loads(geometry).centroid.x, loads(geometry).centroid.y)
                         for geometry in walkways_df['geometry']]

    # Combine all feature positions
    features = crosswalk_positions + lane_positions + walkway_positions
    # Convert the list of features to a tensor with shape (num_features, 2)
    environment_features = torch.tensor(features, dtype=torch.float32)

    # Reshape the tensor to have shape (1, sequence_length, num_features, 2)
    # Repeat the features for each time step in the sequence
    environment_tensor = environment_features.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_features, 2)
    environment_tensor = environment_tensor.repeat(1, sequence_length, 1,
                                                   1)  # Shape: (1, sequence_length, num_features, 2)

    return environment_tensor


def prepare_agent_graph(agent_graph):
    """
    Converts a NetworkX agent graph into tensors for GNN processing.

    Args:
        agent_graph (nx.DiGraph): Directed graph representing agents and interactions.

    Returns:
        node_features (torch.Tensor): Tensor of node features (positions).
        edge_index (torch.Tensor): Tensor of edge indices.
        edge_weights (torch.Tensor): Tensor of edge weights.
    """
    # Extract node features (positions) from the graph
    node_features = torch.tensor([agent_graph.nodes[n]['position'] for n in agent_graph.nodes], dtype=torch.float32)

    # Extract edges and weights
    edge_index = torch.tensor([[u, v] for u, v in agent_graph.edges], dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor([agent_graph[u][v]['weight'] for u, v in agent_graph.edges], dtype=torch.float32)

    return node_features, edge_index, edge_weights


def visualize_scene(crosswalks, lanes, walkways):
    """
    Visualizes the entire scene in a fixed frame, with crosswalks, lanes, and walkways.

    Args:
        crosswalks (list): List of Shapely Polygon objects for crosswalks.
        lanes (list): List of Shapely Polygon objects for lanes.
        walkways (list): List of Shapely Polygon objects for walkways.
    """
    # Create a new plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each crosswalk directly in scene coordinates
    for crosswalk in crosswalks:
        x, y = crosswalk.exterior.xy
        ax.fill(x, y, color='blue', alpha=0.5,
                label='Crosswalk' if 'Crosswalk' not in ax.get_legend_handles_labels()[1] else "")

    # Plot each lane directly in scene coordinates
    for lane in lanes:
        x, y = lane.exterior.xy
        ax.fill(x, y, color='green', alpha=0.5, label='Lane' if 'Lane' not in ax.get_legend_handles_labels()[1] else "")

    # Plot each walkway directly in scene coordinates
    for walkway in walkways:
        x, y = walkway.exterior.xy
        ax.fill(x, y, color='red', alpha=0.5,
                label='Walkway' if 'Walkway' not in ax.get_legend_handles_labels()[1] else "")

    # Set plot limits based on scene dimensions
    ax.set_xlim(min([p.bounds[0] for p in crosswalks + lanes + walkways]),
                max([p.bounds[2] for p in crosswalks + lanes + walkways]))
    ax.set_ylim(min([p.bounds[1] for p in crosswalks + lanes + walkways]),
                max([p.bounds[3] for p in crosswalks + lanes + walkways]))
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Scene-Centered Environment Visualization")
    ax.legend()
    plt.grid(True)
    plt.show()