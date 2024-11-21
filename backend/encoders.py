import torch
import torch.nn as nn


class EnvironmentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EnvironmentEncoder, self).__init__()
        # LSTM layer: transforms input sequence into hidden states
        #   - batch_first=True says the input shape is in (batch, seq, feature) form
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Hidden size is what should be remembered
        self.hidden_size = hidden_size

    def forward(self, env_tensor):
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
        self.gnn = nn.Linear(in_features, out_features)

    def forward(self, agent_data):
        # Put features through the linear layer
        node_embeddings = self.gnn(agent_data)
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
        # 1. Input Processing
        #   -   Get ego and other agents from the tensor
        ego_features = agent_tensor[0]  # Shape is 8 (features)
        agent_features = agent_tensor[1:]   # Shape is 9
        #   -   Get specific positions
        agent_positions = agent_features[:, :2]  # Shape: [num_agents, 2]
        ego_position = ego_features[:2]  # Shape: [2]

        # 2. Calculate influence weights for each agent
        #   -   Distance between agent to ego
        distances = torch.norm(agent_positions - ego_position, dim=1)
        #   -   Calculate base weights - closer agents have more influence
        weights = 1.0 / (distances + 1.0)  # Add 1 to avoid division by zero

        #   -   Extract agent types from features
        is_vehicle = agent_features[:, 6]
        is_pedestrian = agent_features[:, 7]
        is_barrier = agent_features[:, 8]

        #   -   Modify weights based on agent type
        #   --      Vehicles have 5x more influence
        weights = torch.where(is_vehicle == 1, weights * 5.0, weights)
        #   --      Pedestrians have 1x more influence
        weights = torch.where(is_pedestrian == 1, weights * 1.0, weights)
        #   --      Barriers have 1x more influence
        weights = torch.where(is_barrier == 1, weights * 1.0, weights)

        #    -   Normalize weights so they sum to 1
        weights = weights / weights.sum()
        #   -   Add dim for matrix operations
        influence_weights = weights.unsqueeze(1)

        # 3. Feature Transformation
        #   -   Transform all agent features to higher dim
        agent_embeddings = self.fc(agent_features)   # Shape is [num_agents, hidden_dim]
        #   -   Get rid of negatives
        agent_embeddings = self.relu(agent_embeddings)

        # 4. More Weights
        #   -   Weight embeddings by their influence
        weighted_embeddings = agent_embeddings * influence_weights
        #   -   Sum all weighted agent influences into vector
        aggregated_influence = weighted_embeddings.sum(dim=0)   # Shape = [hidden_dim]

        # 5. Final Processing
        #   -   Transform and return final influence vector
        influence_vector = self.fc_aggregate(self.relu(aggregated_influence))

        return influence_vector
