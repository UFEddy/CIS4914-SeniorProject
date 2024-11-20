import pandas as pd
import torch
from directed_graph import createDirectedGraph, directedGraphNodeList, showDirectedGraph
from model import TrajectoryPredictionModel
from utils import encode_environment
import numpy as np


def main():
    # Parameters
    sequence_length = 1
    env_input_size = 2  # x, y coordinates for each feature
    agent_input_size = 8  # x, y coordinates for agents
    hidden_size = 64
    output_size = 2  # predicted x, y coordinates

    # Load and prepare environment data
    print("Loading data...")
    crosswalks_df = pd.read_csv("./data/crosswalks.csv")
    lanes_df = pd.read_csv("./data/lanes_polygons.csv")
    walkways_df = pd.read_csv("./data/walkways.csv")
    ego_pose_df = pd.read_csv("./data/ego_pose.csv")

    print("\nCreating agent tensor...")
    # Create directed graph and get agent tensor
    directed_graph, agent_tensor_data = createDirectedGraph()
    print("Agent tensor shape:", agent_tensor_data.shape)

    # Normalize positions relative to the ego
    x_ref = ego_pose_df['x'].iloc[0]
    y_ref = ego_pose_df['y'].iloc[0]

    ego_pose_df['x_rel'] = ego_pose_df['x'] - x_ref
    ego_pose_df['y_rel'] = ego_pose_df['y'] - y_ref

    # Encode environment
    print("\nEncoding environment...")
    # Get a tensor from the data
    env_tensor = encode_environment(crosswalks_df, lanes_df, walkways_df)

    # Create model
    print("\nInitializing model...")
    model = TrajectoryPredictionModel(
        env_input_size=env_input_size,
        agent_input_size=agent_input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )

    # Prepare ego vehicle trajectory data
    print("\nPreparing trajectory sequences...")
    ego_sequences = []
    for i in range(len(ego_pose_df) - sequence_length):
        sequence = ego_pose_df.iloc[i:i + sequence_length][['x_rel', 'y_rel']].values
        target = ego_pose_df.iloc[i + sequence_length][['x_rel', 'y_rel']].values
        ego_sequences.append((sequence, target))

    # Make predictions
    print("\nMaking predictions...")
    model.eval()
    with torch.no_grad():
        # Test on first sequence
        sequence_tensor = torch.FloatTensor(ego_sequences[0][0]).unsqueeze(0)
        target_tensor = torch.FloatTensor(ego_sequences[0][1])

        print("Input sequence shape:", sequence_tensor.shape)
        prediction = model(env_tensor, agent_tensor_data)

        print("\nPrediction results:")
        print("Target relative position:", target_tensor.numpy())
        print("Predicted relative position:", prediction.squeeze().numpy())

        # Convert predictions back to global coordinates
        target_global = target_tensor.numpy() + [x_ref, y_ref]
        prediction_global = prediction.squeeze().numpy() + [x_ref, y_ref]

        print("\nGlobal coordinates:")
        print("Target position:", target_global)
        print("Predicted position:", prediction_global)

        # Calculate error
        prediction_error = np.linalg.norm(target_tensor.numpy() - prediction.squeeze().numpy())
        print("\nPrediction error (Euclidean distance):", prediction_error)


if __name__ == '__main__':
    main()
