import torch
import torch.nn as nn
from encoders import EnvironmentEncoder, AgentInteractionEncoder, InfluenceEncoder


class TrajectoryPredictionModel(nn.Module):
    def __init__(self, env_input_size, agent_input_size, hidden_size, output_size):
        super(TrajectoryPredictionModel, self).__init__()

        # Environment encoder: processes road/scene information
        #   - Takes in environment data and learns what's important about it
        self.env_encoder = EnvironmentEncoder(env_input_size, hidden_size)

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

        # Final layer: combines everything to predict where we'll go
        #   - Takes all processed info and predicts our next position
        self.fc = nn.Linear(hidden_size * 3, output_size)

    def forward(self, env_tensor, agent_tensor):
        # Turn environment tensor into useful information
        env_encoding = self.env_encoder(env_tensor)

        # Turn agent tensor into useful information
        #   - Like looking at each car/person and remembering key things
        agent_encoding = self.agent_encoder(agent_tensor)
        # Put all agent information into one "chunk"
        #   - Instead of remembering 10 different cars, make one summary
        agent_encoding = torch.mean(agent_encoding, dim=0, keepdim=True)

        # Figure out how nearby cars/people affect the decision
        influence_encoding = self.influence_encoder(agent_tensor)
        # Add batch dimension so shapes match up for combining
        influence_encoding = influence_encoding.unsqueeze(0)

        # Combine all the pieces
        combined = torch.cat((env_encoding, agent_encoding, influence_encoding), dim=-1)

        # Use everything we know to predict where to drive
        output = self.fc(combined)
        return output
