import torch
import torch.nn as nn

# https://pytorch.org/docs/stable/generated/torch.norm.html
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html


class EnvironmentEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(EnvironmentEncoder, self).__init__()

        # Separate encoders for different environment features
        self.lane_encoder = nn.Sequential(
            nn.Linear(7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.crosswalk_encoder = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.walkway_encoder = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Attention mechanism for each feature type
        self.lane_attention = nn.Linear(hidden_size, 1)
        self.crosswalk_attention = nn.Linear(hidden_size, 1)
        self.walkway_attention = nn.Linear(hidden_size, 1)

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.spatial_radius = 50.0  # meters

    def forward(self, environment_features, ego_position):
        """
        Args:
            environment_features: Tuple of (lane_features, crosswalk_features, walkway_features)
                Each feature is a tensor containing the processed geometric data
            ego_position: tensor of shape (batch_size, 2) containing (x, y) coordinates
        """
        lane_features, crosswalk_features, walkway_features = environment_features
        batch_size = ego_position.shape[0]
        device = ego_position.device

        # Handle empty features case
        if len(lane_features) == 0:
            lane_features = torch.zeros((batch_size, 1, 7)).to(device)
        if len(crosswalk_features) == 0:
            crosswalk_features = torch.zeros((batch_size, 1, 5)).to(device)
        if len(walkway_features) == 0:
            walkway_features = torch.zeros((batch_size, 1, 4)).to(device)

        # Ensure all features are on the correct device
        lane_features = lane_features.to(device)
        crosswalk_features = crosswalk_features.to(device)
        walkway_features = walkway_features.to(device)

        # Add batch dimension if not present
        if len(lane_features.shape) == 2:
            lane_features = lane_features.unsqueeze(0).expand(batch_size, -1, -1)
        if len(crosswalk_features.shape) == 2:
            crosswalk_features = crosswalk_features.unsqueeze(0).expand(batch_size, -1, -1)
        if len(walkway_features.shape) == 2:
            walkway_features = walkway_features.unsqueeze(0).expand(batch_size, -1, -1)

        # Process lanes
        lane_encoded = self.lane_encoder(lane_features)
        lane_attention = torch.softmax(self.lane_attention(lane_encoded), dim=1)
        lane_context = (lane_attention * lane_encoded).sum(dim=1)

        # Process crosswalks
        crosswalk_encoded = self.crosswalk_encoder(crosswalk_features)
        crosswalk_attention = torch.softmax(self.crosswalk_attention(crosswalk_encoded), dim=1)
        crosswalk_context = (crosswalk_attention * crosswalk_encoded).sum(dim=1)

        # Process walkways
        walkway_encoded = self.walkway_encoder(walkway_features)
        walkway_attention = torch.softmax(self.walkway_attention(walkway_encoded), dim=1)
        walkway_context = (walkway_attention * walkway_encoded).sum(dim=1)

        # Combine contexts
        combined_context = torch.cat([lane_context, crosswalk_context, walkway_context], dim=1)

        # Final fusion
        environment_encoding = self.fusion_layer(combined_context)

        return environment_encoding


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
