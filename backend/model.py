import torch
import torch.nn as nn
from encoders import EnvironmentEncoder, AgentInteractionEncoder, InfluenceEncoder


class TrajectoryPredictionModel(nn.Module):
    def __init__(self, agent_input_size, hidden_size, output_size):
        super(TrajectoryPredictionModel, self).__init__()

        # Environment encoder: processes road/scene information
        #   - Takes in environment data and learns what's important about it
        self.env_encoder = EnvironmentEncoder(hidden_size)

        # Agent encoder: processes all the cars/people around us
        #   - Takes raw agent features and makes them easier to understand
        self.agent_encoder = AgentInteractionEncoder(
            in_features=agent_input_size,
            out_features=hidden_size
        )

        # Influence encoder: figures out how each agent affects our car
        #   - Helps understand which agents we should pay more attention to
        self.influence_encoder = InfluenceEncoder(
            input_dim=agent_input_size,
            hidden_dim=hidden_size
        )

        # Output layers with added complexity
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, environment_features, ego_position, agent_tensor):
        # Turn environment tensor into useful information
        env_encoding = self.env_encoder(environment_features, ego_position)

        # Turn agent tensor into useful information
        #   - Like looking at each car/person and remembering key things
        agent_encoding = self.agent_encoder(agent_tensor)
        # Put all agent information into one "chunk"
        #   - Instead of remembering 10 different cars, make one summary
        agent_encoding = torch.mean(agent_encoding, dim=1)

        # Influence encoding - [batch_size, hidden_size]
        influence_encoding = self.influence_encoder(agent_tensor)

        # Ensure all tensors have shape [batch_size, hidden_size]
        # if len(env_encoding.shape) != 2:
        #     env_encoding = env_encoding.squeeze(1)
        if len(agent_encoding.shape) != 2:
            agent_encoding = agent_encoding.squeeze(1)
        if len(influence_encoding.shape) != 2:
            influence_encoding = influence_encoding.squeeze(1)

        # Combine all encodings
        combined = torch.cat((env_encoding, agent_encoding, influence_encoding), dim=1)

        # Generate prediction
        output = self.combined_layer(combined)
        output = output * 10.0  # Scale to ±10 meters
        return output
