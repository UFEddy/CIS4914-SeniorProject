import pandas as pd
import torch
from directed_graph import createDirectedGraph, directedGraphNodeList, showDirectedGraph
from model import TrajectoryPredictionModel
from utils import encode_environment
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Parameters
    sequence_length = 10
    env_input_size = 2  # x, y coordinates for each feature
    agent_input_size = 9  # x, y coordinates for agents
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
    print("\nAgent tensor shape:", agent_tensor_data.shape)

    # Get current position (from first row)
    x_current = ego_pose_df['x'].iloc[1]
    y_current = ego_pose_df['y'].iloc[1]

    # Get actual future position (next position in dataset)
    x_actual = ego_pose_df['x'].iloc[0]
    y_actual = ego_pose_df['y'].iloc[0]

    # Normalize positions relative to the ego
    ego_pose_df['x_rel'] = ego_pose_df['x'] - x_current
    ego_pose_df['y_rel'] = ego_pose_df['y'] - y_current

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

        prediction_global = prediction.squeeze().numpy() + [x_current, y_current]

        print("\nTrajectory Comparison:")
        print(f"Current position: ({x_current:.2f}, {y_current:.2f})")
        print(f"Actual next position: ({x_actual:.2f}, {y_actual:.2f})")
        print(f"Predicted next position: ({prediction_global[0]:.2f}, {prediction_global[1]:.2f})")

        # Calculate prediction error
        prediction_error = np.sqrt(
            (prediction_global[0] - x_actual) ** 2 +
            (prediction_global[1] - y_actual) ** 2
        )
        print(f"\nPrediction error (meters): {prediction_error:.2f}")

        # Visualize trajectories
        # plt.figure(figsize=(10, 10))
        #
        # # Plot positions
        # plt.scatter(x_current, y_current, c='blue', s=100, label='Current Position')
        # plt.scatter(x_actual, y_actual, c='green', s=100, label='Actual Next Position')
        # plt.scatter(prediction_global[0], prediction_global[1], c='red', s=100, label='Predicted Next Position')
        #
        # # Draw trajectories
        # plt.plot([x_current, x_actual], [y_current, y_actual], 'g--', alpha=0.5, label='Actual Path')
        # plt.plot([x_current, prediction_global[0]], [y_current, prediction_global[1]], 'r--', alpha=0.5,
        #          label='Predicted Path')
        #
        # plt.title('Ego Vehicle Trajectory: Prediction vs Actual')
        # plt.xlabel('X Position (m)')
        # plt.ylabel('Y Position (m)')
        # plt.legend()
        # plt.grid(True)
        # plt.axis('equal')
        # plt.show()
        #
        # # Show agent positions relative to ego
        # plt.figure(figsize=(10, 10))


if __name__ == '__main__':
    main()
