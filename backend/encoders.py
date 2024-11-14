# encoders.py
import torch
import torch.nn as nn


class EnvironmentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnvironmentEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, env_tensor):
        _, (hn, _) = self.lstm(env_tensor)  # Only keep the last hidden state
        return hn[-1]  # Shape: (batch_size, hidden_size)


class AgentInteractionEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(AgentInteractionEncoder, self).__init__()
        self.gnn = nn.Linear(in_features, out_features)

    def forward(self, agent_graph):
        # Convert agent graph to input tensor for GNN
        node_features = torch.tensor([agent_graph.nodes[n]['position'] for n in agent_graph.nodes])
        edge_indices = torch.tensor([[u, v] for u, v in agent_graph.edges])
        edge_weights = torch.tensor([agent_graph[u][v]['weight'] for u, v in agent_graph.edges])

        # Apply GNN layer (simplified)
        node_embeddings = self.gnn(node_features)
        return node_embeddings  # Shape: (num_agents, out_features)
