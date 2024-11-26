import torch
import torch.nn as nn

# https://pytorch.org/docs/stable/generated/torch.norm.html
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html

class EnvironmentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnvironmentEncoder, self).__init__()
        # LSTM layer: transforms input sequence into hidden states
        #   - batch_first=True says the input shape is in (batch, seq, feature) form
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Hidden size is what should be remembered
        self.hidden_size = hidden_size

    def forward(self, env_tensor):
        # Ensure data type for tensor
        env_tensor = env_tensor.to(torch.float32)
        # Get batch size from input tensor (how many batches to produce)
        batch_size = env_tensor.size(0)
        # Create initial hidden state with zeros
        #   - shape: [1, batch_size, hidden_size]
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(env_tensor.device)
        # Create initial cell state with zeros, same shape
        #   - cells contains information about the environment
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(env_tensor.device)

        # Run LSTM: process the environment features through a lstm layer
        #   - output contains all hidden states, hidden contains final states
        output, (hidden, _) = self.lstm(env_tensor, (h0, c0))

        # Return last hidden state
        return hidden[-1]


class AgentInteractionEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(AgentInteractionEncoder, self).__init__()
        # Linear layer: transform agent features into higher dimension
        #   maps from in_features to out_features for richer representation
        self.gnn = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, agent_data):
        # Ensure input is 3D: [batch_size, num_agents, features]
        agent_data = agent_data.to(torch.float32)
        if len(agent_data.shape) == 2:
            agent_data = agent_data.unsqueeze(0)

        batch_size, num_agents, _ = agent_data.shape

        # Process each agent's features
        node_embeddings = self.gnn(agent_data)  # [batch_size, num_agents, out_features]
        return node_embeddings

      
class InfluenceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(InfluenceEncoder, self).__init__()
        # Linear layer: transforms raw agent features to hidden representation
        #   - Data is taken in and mixes them up in different ways
        self.fc = nn.Linear(input_dim, hidden_dim)
        # Linear layer: computes attention scores for each agent
        self.attention = nn.Linear(hidden_dim, 1)
        # ReLU: adds non-linearity to transformations
        #   ReLu (Rectified Linear Unit)
        #   -   Gets rid of negative numbers
        self.relu = nn.ReLU()
        # Linear layer: final transformation of aggregated features
        self.fc_aggregate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, agent_tensor):
        # Handle batch dimension
        # Ensure float32
        agent_tensor = agent_tensor.to(torch.float32)
        # If agent_tensor is [batch_size, num_agents, features]
        batch_size = agent_tensor.size(0)

        # Process each batch separately
        batch_influences = []

        for b in range(batch_size):
            # Get current batch data
            current_batch = agent_tensor[b]  # [num_agents, features]

            # Extract ego and agent features
            ego_features = current_batch[0]  # [features]
            agent_features = current_batch[1:]  # [num_agents-1, features]

            if len(agent_features) == 0:
                # Handle case with no agents
                return torch.zeros(batch_size, self.fc.out_features).to(agent_tensor.device)

            # Get positions
            agent_positions = agent_features[:, :2]  # First 2 features are x,y positions
            ego_position = ego_features[:2]

            # Calculate distances
            distances = torch.norm(agent_positions - ego_position.unsqueeze(0), dim=1)
            weights = 1.0 / (distances + 1.0)

            # Agent type weights
            is_vehicle = agent_features[:, 6]
            is_pedestrian = agent_features[:, 7]
            is_barrier = agent_features[:, 8]

            # Modify weights
            weights = torch.where(is_vehicle == 1, weights * 5.0, weights)
            weights = torch.where(is_pedestrian == 1, weights * 1.0, weights)
            weights = torch.where(is_barrier == 1, weights * 1.0, weights)

            # Normalize weights
            weights = weights / (weights.sum() + 1e-10)  # Added small epsilon for numerical stability
            influence_weights = weights.unsqueeze(1)

            # Transform features
            agent_embeddings = self.fc(agent_features)  # [num_agents-1, hidden_dim]
            agent_embeddings = self.relu(agent_embeddings)

            # Apply weights
            weighted_embeddings = agent_embeddings * influence_weights
            aggregated_influence = weighted_embeddings.sum(dim=0)  # [hidden_dim]

            # Final transformation
            influence_vector = self.fc_aggregate(self.relu(aggregated_influence))
            batch_influences.append(influence_vector)

        # Stack all batch influences
        return torch.stack(batch_influences)
