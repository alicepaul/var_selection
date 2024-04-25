import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def generate_max_cut_problem(num_vertices, edge_prob=0.5, weight_range=(1, 10)):
    """
    Generates a random Erdős-Rényi graph for the Max Cut problem.

    Parameters:
        num_vertices (int): Number of vertices in the graph.
        edge_prob (float): Probability of an edge between any two nodes.
        weight_range (tuple): A tuple (min_weight, max_weight) defining the range
                              of weights for the edges.

    Returns:
        np.array: An adjacency matrix representing the generated graph, where
                  matrix[i][j] contains the weight of the edge between vertices
                  i and j; 0 if no edge exists.
    """
    # Create an Erdős-Rényi graph
    G = nx.erdos_renyi_graph(num_vertices, edge_prob)

    # Initialize an adjacency matrix with zeros
    adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    # Assign random weights to the edges
    for (u, v) in G.edges():
        weight = random.randint(*weight_range)
        adjacency_matrix[u][v] = weight
        adjacency_matrix[v][u] = weight  # The graph is undirected

    return adjacency_matrix

num_vertices = 25
edge_prob = 0.5  # Probability of an edge
weight_range = (1, 10)
problem_set = generate_max_cut_problem(num_vertices, edge_prob, weight_range)


def save_graphs(num_graphs, num_vertices, edge_prob, weight_range, directory):
    """
    Generates a specified number of graphs and saves each graph's adjacency matrix to a separate text file.

    Parameters:
        num_graphs (int): Number of graphs to generate.
        num_vertices (int): Number of vertices in each graph.
        edge_prob (float): Probability of an edge between any two nodes.
        weight_range (tuple): A tuple (min_weight, max_weight) defining the range of weights for the edges.
        directory (str): Path to the directory where text files should be saved.
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(num_graphs):
        adjacency_matrix = generate_max_cut_problem(num_vertices, edge_prob, weight_range)
        file_path = os.path.join(directory, f'graph_{i+1}.txt')
        np.savetxt(file_path, adjacency_matrix, fmt='%d')

save_graphs(60, 30, 0.25, (1, 10), "maxcut-tests/30vars_0.25")

## generated = 25 with 0.4, 35 with 0.2, 30 with 0.25