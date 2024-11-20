from cdlib import algorithms
from cdlib.classes.node_clustering import NodeClustering
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re

def visualize_graph(G, with_labels=True, node_color='lightblue', node_size=700, font_size=12, pos = None):
    """
    Visualize a NetworkX graph.

    Parameters:
    G (networkx.Graph): The NetworkX graph to visualize.
    with_labels (bool): If True, nodes will be labeled with their ids.
    node_color (str): Color of the nodes.
    node_size (int): Size of the nodes.
    font_size (int): Font size for node labels.
    """
    plt.figure(figsize=(12, 8))  # Set the size of the plot
    # Use the spring layout if no position is given
    if pos is None:
        pos = nx.spring_layout(G)
    
    # Draw the graph
    nx.draw(G, pos, with_labels=with_labels, node_color=node_color, node_size=node_size, font_size=font_size)

    # Draw edge labels (optional, uncomment if needed)
    # edge_labels = nx.get_edge_attributes(G, 'label')  # Assuming 'label' is the attribute name
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

def visualize_communities(G, communities, pos=None, node_size=100, font_size=8, with_labels=False):
    """
    Visualize communities in a network.

    Parameters:
    G (networkx.Graph): A NetworkX graph.
    communities (cdlib.classes.node_clustering.NodeClustering): Communities detected by CDlib.
    pos (dict, optional): Positions of nodes for layout.
    node_size (int, optional): Size of nodes.
    font_size (int, optional): Font size for node labels.
    with_labels (bool, optional): If True, nodes will be labeled with their ids.
    """
    # Generate a color palette with enough colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities.communities)))

    # Use the spring layout if no position is given
    if pos is None:
        pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 8))

    # Draw nodes and edges
    for i, community in enumerate(communities.communities):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[colors[i]]*len(community), node_size=node_size)
    nx.draw_networkx_edges(G, pos)

    # Optionally add labels
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size)

    plt.title("Community Detection")
    plt.show()
    
def visualize_overlapping_communities(G, node_clustering, pos=None, node_size=100, font_size=8, with_labels=False):
    """
    Visualize overlapping communities in a network.

    Parameters:
    G (networkx.Graph): A NetworkX graph.
    node_clustering (cdlib.classes.node_clustering.NodeClustering): NodeClustering object from CDlib representing communities.
    pos (dict, optional): Positions of nodes for layout.
    node_size (int, optional): Size of nodes.
    font_size (int, optional): Font size for node labels.
    with_labels (bool, optional): If True, nodes will be labeled with their ids.
    """
    # Convert NodeClustering to a list of sets
    communities = [set(community) for community in node_clustering.communities]

    # Generate a color palette with enough colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))

    # Create a mapping of node to list of colors (one for each community it belongs to)
    node_colors = {node: [] for node in G.nodes()}
    for i, community in enumerate(communities):
        for node in community:
            node_colors[node].append(colors[i])

    # Compute the mixed color for each node (average of community colors)
    for node in node_colors:
        color_array = np.array(node_colors[node])
        mixed_color = np.mean(color_array, axis=0)
        node_colors[node] = mixed_color

    # Use the spring layout if no position is given
    if pos is None:
        pos = nx.spring_layout(G)

    plt.figure(figsize=(12, 8))

    # Draw nodes
    for node in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=[node_colors[node]], node_size=node_size)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Optionally add labels
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=font_size)

    plt.title("Overlapping Community Detection")
    plt.show()