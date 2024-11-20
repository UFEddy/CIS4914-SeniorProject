import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List
from networkx.algorithms.bipartite.basic import color
import pandas as pd

# Node Superclass
class Node:
    def __init__(self, name, x_global_position, y_global_position, x_velocity, y_velocity):
        self.name = name
        self.x_global_position = x_global_position
        self.y_global_position = y_global_position
        self.x_velocity = x_velocity
        self.y_velocity = y_velocity


# Ego Node Subclass
class EgoNode(Node):
    def __init__(self, name, x_global_position, y_global_position, x_velocity, y_velocity,
                 x_acceleration, y_acceleration, x_angular_rate, y_angular_rate):
        super().__init__(name, x_global_position, y_global_position, x_velocity, y_velocity)
        self.x_acceleration = x_acceleration
        self.y_acceleration = y_acceleration
        self.x_angular_rate = x_angular_rate
        self.y_angular_rate = y_angular_rate
        self.node_type = 'Ego'


# Agent Node Subclass
class AgentNode(Node):
    def __init__(self, name, x_global_position, y_global_position, x_velocity, y_velocity,
                 yaw, category):
        super().__init__(name, x_global_position, y_global_position, x_velocity, y_velocity)
        self.yaw = yaw
        self.category = category
        self.node_type = 'Agent'


# Dictionary for agent weights based on their category - Weights are static
agentCategoryToWeightDictionary = {
    'pedestrian': 1,
    'vehicle': 5,
    'barrier': 1
}

# Dictionary for agent colors - For color_map only
agentColorDictionary = {
    'pedestrian': 'blue',
    'vehicle': 'red',
    'barrier': 'gray'
}

# List to hold nodes for all following operations
nodeList = []
# Contains colors for each node
color_map = []


# Creates and returns the directed graph
def createDirectedGraph():
    # TODO - These nodes will want to be from the dataset with all their appropriate values

    #Ego = EgoNode('Ego', 10, 10, 0, 0, 0, 0, 0, 0)
    #Agent1 = AgentNode('Agent1', 2, 4, 0, 0, 0, 0, 0, 'pedestrian')
    #Agent2 = AgentNode('Agent2', 6, 4, 0, 0, 0, 0, 0, 'vehicle')

    # Load in data
    agent_data = pd.read_csv('./data/agent_data_10secs.csv')
    # Standardize category names to lowercase
    agent_data['category'] = agent_data['category'].str.lower()
    
    # Exclude traffic cone and generic object categories from agent data
    agent_data = agent_data[~agent_data['category'].isin(['traffic_cone', 'generic_object'])]

    # Create Ego nodes
    ego_data = pd.read_csv('./data/ego_pose.csv')
    # Only taking the last position (top of file)
    ego_row = ego_data.iloc[0]
    ego_node = EgoNode(
        name='Ego0',
        x_global_position=ego_row['x'],
        y_global_position=ego_row['y'],
        x_velocity=ego_row['vx'],
        y_velocity=ego_row['vy'],
        x_acceleration=ego_row['acceleration_x'],
        y_acceleration=ego_row['acceleration_y'],
        x_angular_rate=ego_row['angular_rate_x'],
        y_angular_rate=ego_row['angular_rate_y']
    )
    nodeList.append(ego_node)

    # Create Agent nodes
    for index, row in agent_data.iterrows():
        agent = AgentNode(
            name=f'Agent{index+1}',
            x_global_position=row['x'],
            y_global_position=row['y'],
            x_velocity=row['vx'],
            y_velocity=row['vy'],
            yaw=row['yaw'],
            category=row['category']
        )
        nodeList.append(agent)

    # TODO - Don't forget to append the above nodes to the node list
    #nodeList.append(Ego)
    #nodeList.append(Agent1)
    #nodeList.append(Agent2)


    # Initializes directed graph
    directed_graph = nx.DiGraph()
    # Initializes position layout
    pos = {}

    # Lists to build tensor data
    features_list = []

    # Adds node positions and edges
    # TODO - Colors aren't being set to the correct nodes
    for node in nodeList:
        pos[node.name] = np.array([node.x_global_position, node.y_global_position])

        # Only adds edges for agent nodes, as all edges point to the ego node
        if node.node_type == 'Agent':
            # Tensor data for features
            features = [
                node.x_global_position,
                node.y_global_position,
                node.x_velocity,
                node.y_velocity,
                np.cos(node.yaw), # Encode yaw as sin/cos
                np.sin(node.yaw),
                float(node.category == 'vehicle'),
                float(node.category == 'pedestrian')
            ]
            features_list.append(features)

            # NetworkX Graph
            directed_graph.add_edge(node.name, 'Ego0', weight=agentCategoryToWeightDictionary[node.category])
            color_map.append(agentColorDictionary[node.category])

        else:
            # Tensor data for ego node
            features = [
                node.x_global_position,
                node.y_global_position,
                node.x_velocity,
                node.y_velocity,
                1.0,  # cos(0) for yaw
                0.0,  # sin(0) for yaw
                0.0,  # not a vehicle
                0.0   # not a pedestrian
            ]
            features_list.append(features)

            # NetworkX Graph
            color_map.append('pink')

    # Create the tensor data
    agent_tensor = torch.tensor(features_list, dtype=torch.float32)

    # Draw the graph
    # plt.figure(figsize=(12, 8))
    # nx.draw(
    #     directed_graph, pos, edge_color='black',
    #     width=1, linewidths=1,
    #     node_size=500, node_color=color_map,
    #     alpha=0.9,
    #     labels={node: node for node in directed_graph.nodes()}
    # )
    #
    # # Edge labels
    # for node in nodeList:
    #     pos[node.name] = np.array([node.x_global_position, node.y_global_position])
    #     if node.name != 'Ego0':
    #         nx.draw_networkx_edge_labels(
    #             directed_graph, pos,
    #             edge_labels={(node.name, 'Ego'): 'Weight: ' + str(agentCategoryToWeightDictionary[node.category])}
    #         )
    # plt.axis('off')

    return directed_graph, agent_tensor


# Returns the node list of the directed graph
def directedGraphNodeList():
    return nodeList


# Shows the directed graph
def showDirectedGraph():
    plt.show()
