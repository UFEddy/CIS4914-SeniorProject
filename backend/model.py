# model.py
import torch
import torch.nn as nn
from encoders import EnvironmentEncoder, AgentInteractionEncoder


class TrajectoryPredictionModel(nn.Module):
    def __init__(self, env_input_size, agent_input_size, hidden_size, output_size):
        super(TrajectoryPredictionModel, self).__init__()

        # Environment encoder
        self.env_encoder = EnvironmentEncoder(env_input_size, hidden_size)

        # Agent encoder
        self.agent_encoder = AgentInteractionEncoder(agent_input_size, hidden_size)

        # Layer to combine environment and agent
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, env_tensor, agent_graph):
        # Encode environment tensor with LSTM
        env_encoding = self.env_encoder(env_tensor)

        # Encode agent graph with GNN layer
        agent_encoding = self.agent_encoder(agent_graph)

        # Combine both encodings
        combined = torch.cat((env_encoding, agent_encoding.mean(dim=0)), dim=-1)

        # Predict future position
        output = self.fc(combined)
        return output  # Shape: (batch_size, output_size)
