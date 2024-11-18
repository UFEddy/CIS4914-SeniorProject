import torch
import torch.nn as nn


class EnvironmentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnvironmentEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, env_tensor):
        # Initialize hidden state
        batch_size = env_tensor.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(env_tensor.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(env_tensor.device)

        # Process with LSTM
        output, (hidden, _) = self.lstm(env_tensor, (h0, c0))

        # Return last hidden state
        return hidden[-1]


class AgentInteractionEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(AgentInteractionEncoder, self).__init__()
        self.gnn = nn.Linear(in_features, out_features)

    def forward(self, agent_data):
        # For testing - assume agent_data is just a tensor
        # Shape: (batch_size, features)
        node_embeddings = self.gnn(agent_data)
        return node_embeddings
