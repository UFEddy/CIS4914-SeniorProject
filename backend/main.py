import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from directed_graph import createDirectedGraph
from model import TrajectoryPredictionModel
import numpy as np
import datetime

from utils import prepare_environment_data, process_environment_data


def main():
    # Parameters
    sequence_length = 10
    env_input_size = 2
    agent_input_size = 9
    hidden_size = 64
    output_size = 2
    batch_size = 32
    num_epochs = 25  # Update to for additional trainings, 2 is min to see if it's working
    learning_rate = 0.1

    # Load data
    print("Loading data...")
    crosswalks_df = pd.read_csv("./data/crosswalks.csv")
    lanes_df = pd.read_csv("./data/lanes_polygons.csv")
    walkways_df = pd.read_csv("./data/walkways.csv")
    ego_pose_df = pd.read_csv("./data/ego_pose.csv")

    # Create agent tensor
    print("\nCreating agent tensor...")
    directed_graph, agent_tensor = createDirectedGraph()
    # Convert agent tensor to float32
    agent_tensor = agent_tensor.to(torch.float32)
    print("Agent tensor shape:", agent_tensor.shape)

    # Create sequences
    print("\nPreparing trajectory sequences...")
    sequences = []
    targets = []

    for i in range(len(ego_pose_df) - sequence_length):
        sequence = ego_pose_df.iloc[i:i + sequence_length][['x', 'y']].values.astype(np.float32)
        target = ego_pose_df.iloc[i + sequence_length][['x', 'y']].values.astype(np.float32)
        sequences.append(sequence)
        targets.append(target)

    # Convert to tensors efficiently
    sequences_array = np.array(sequences)
    targets_array = np.array(targets)

    # Normalize data
    pos_mean = ego_pose_df[['x', 'y']].mean()
    pos_std = ego_pose_df[['x', 'y']].std()
    # Convert mean and std to simple numpy arrays
    pos_mean_x = pos_mean['x'].item()  # Convert to Python scalar
    pos_mean_y = pos_mean['y'].item()
    pos_std_x = pos_std['x'].item()
    pos_std_y = pos_std['y'].item()

    pos_mean = np.array([pos_mean_x, pos_mean_y], dtype=np.float32)
    pos_std = np.array([pos_std_x, pos_std_y], dtype=np.float32)

    # Normalize using numpy arrays
    sequences_tensor = torch.FloatTensor(sequences_array)
    targets_tensor = torch.FloatTensor(targets_array)

    sequences_tensor = (sequences_tensor - torch.from_numpy(pos_mean)) / torch.from_numpy(pos_std)
    targets_tensor = (targets_tensor - torch.from_numpy(pos_mean)) / torch.from_numpy(pos_std)

    # Encode environment
    print("\nEncoding environment...")
    # Prepare the environment data
    lanes_df, crosswalks_df, walkways_df = prepare_environment_data(
        crosswalks_df, lanes_df, walkways_df
    )

    # Initialize model and training components
    model = TrajectoryPredictionModel(
        agent_input_size=agent_input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )

    # Convert model parameters to float32
    model = model.to(torch.float32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    print("\nStarting training...")
    losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        # TRAINING LOOP
        for i in range(0, len(sequences_tensor), batch_size):
            batch_sequences = sequences_tensor[i:i + batch_size]
            batch_targets = targets_tensor[i:i + batch_size]

            current_positions = batch_sequences[:, -1, :]  # Last position in sequence
            optimizer.zero_grad()

            # Prepare input tensors
            environment_features = process_environment_data(
                crosswalks_df, lanes_df, walkways_df, current_positions
            )
            agent_tensor_batch = agent_tensor.unsqueeze(0).expand(len(batch_sequences), -1, -1)

            predictions = model(environment_features, current_positions, agent_tensor_batch)
            loss = criterion(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)

        now = datetime.datetime.now()
        print(f"{now.time().strftime('%H:%M:%S')}: Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        # Early stopping check
        if epoch > 10 and losses[-1] > losses[-2] > losses[-3]:
            print("Early stopping triggered")
            break

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    # Model evaluation
    model.eval()
    with torch.no_grad():
        test_sequence = sequences_tensor[-1].unsqueeze(0)
        test_agent_tensor = agent_tensor.unsqueeze(0)
        test_position = test_sequence[:, -1, :]  # Last position of sequence

        environment_features = process_environment_data(
            crosswalks_df,
            lanes_df,
            walkways_df,
            test_position
        )
        test_prediction = model(environment_features, test_position, test_agent_tensor)

        # Denormalize predictions
        prediction_denorm = (test_prediction.squeeze().numpy() * pos_std) + pos_mean
        actual_denorm = (targets_tensor[-1].numpy() * pos_std) + pos_mean

        print("\nPrediction Results:")
        print(f"Predicted position: ({prediction_denorm[0]:.2f}, {prediction_denorm[1]:.2f})")
        print(f"Actual position: ({actual_denorm[0]:.2f}, {actual_denorm[1]:.2f})")

        prediction_error = np.linalg.norm(prediction_denorm - actual_denorm)
        print(f"Prediction error (meters): {prediction_error:.2f}")


if __name__ == '__main__':
    main()