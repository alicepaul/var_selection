from pyscipopt import Model, quicksum
import numpy as np

class Problem:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = len(graph)
        self.gap_tol = 1e-4
        self.prob_stats, self.var_stats = self.get_static_stats()

    def lower_solve(self, node):
        model = Model("MaxCutRelaxed")
        x = {i: model.addVar(vtype="C", lb=0, ub=1, name=f"x_{i}") for i in range(self.num_nodes)}
        y = {(i, j): model.addVar(vtype="C", lb=0, ub=1, name=f"y_{i}_{j}") for i in range(self.num_nodes) for j in range(i+1, self.num_nodes)}
        obj = quicksum(self.graph[i][j] * y[i, j] for i in range(self.num_nodes) for j in range(i+1, self.num_nodes))
        model.setObjective(obj, "maximize")
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                model.addCons(y[i, j] <= x[i] + x[j])
                model.addCons(y[i, j] <= 2 - x[i] - x[j])
                model.addCons(y[i, j] >= x[i] - x[j])
                model.addCons(y[i, j] >= x[j] - x[i])
        for i in node.zlb:
            model.addCons(x[i] == 0)
        for i in node.zub:
            model.addCons(x[i] == 1)
        print("lower_solve node.zlb node.zub", node.zlb, node.zub)
        model.setIntParam("display/verblevel", 1)
        # model.setParam("presolving/maxrounds", 0)
        # model.setParam("presolving/maxrestarts", 0)
        model.optimize()
        if model.getStatus() == "infeasible":
            return -np.inf, -np.inf
        print(model.getObjVal())
        primal_value = model.getPrimalbound()
        dual_value = model.getDualbound()
        node.primal_value = primal_value
        node.dual_value = dual_value
        node.primal_beta = [model.getVal(x[i]) for i in range(self.num_nodes)]
        node.z = [model.getVal(x[i]) for i in range(self.num_nodes)]
        node.support = [i for i in range(self.num_nodes) if i not in node.zlb and i not in node.zub]
        return primal_value, dual_value
    
    def upper_solve(self, node):
        print("upper_solve node.z", node.z)
        partition = [1 if z >= 0.5 else 0 for z in node.z]  # Simplified rounding based on 0.5 threshold

        cut_value = sum(self.graph[i][j] for i in range(self.num_nodes) 
                        for j in range(i + 1, self.num_nodes) if partition[i] != partition[j])

        node.upper_bound = cut_value
        node.upper_z = partition
        print("upper_solve partition", partition)

        return cut_value
    
    def get_static_stats(self):
        """
        Computes and returns static statistics related to the max cut problem.

        Returns:
            tuple: A tuple containing two NumPy arrays:
                - prob_stats: Problem-level statistics.
                - var_stats: Variable-level statistics.
        """
        # Initialize arrays to collect statistics
        degrees = np.zeros(self.num_nodes)
        edge_weights = []

        # Iterate through the graph to collect statistics
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.graph[i][j] != 0:
                    degrees[i] += 1
                    degrees[j] += 1
                    edge_weights.append(self.graph[i][j])
        
        edge_weights = np.array(edge_weights)
        total_weight = np.sum(edge_weights)
        average_weight = np.mean(edge_weights) if edge_weights.size > 0 else 0
        max_weight = np.max(edge_weights) if edge_weights.size > 0 else 0
        min_weight = np.min(edge_weights) if edge_weights.size > 0 else 0
        weight_std_dev = np.std(edge_weights) if edge_weights.size > 0 else 0

        # Calculate graph density
        num_edges = len(edge_weights)
        max_possible_edges = self.num_nodes * (self.num_nodes - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0

        # Additional statistics
        max_degree = np.max(degrees)
        min_degree = np.min(degrees)
        isolated_nodes = np.sum(degrees == 0)

        # Average path length and clustering coefficient are more complex and can be expensive to compute for large graphs.
        # Here we provide a simple approach for clustering coefficient:
        clustering_coefficients = []
        for i in range(self.num_nodes):
            neighbors = [j for j in range(self.num_nodes) if self.graph[i][j] > 0]
            if len(neighbors) > 1:
                actual_links = sum(1 for x in neighbors for y in neighbors if x != y and self.graph[x][y] > 0)
                possible_links = len(neighbors) * (len(neighbors) - 1) / 2
                clustering_coefficients.append(actual_links / possible_links)
            else:
                clustering_coefficients.append(0)
        average_clustering_coefficient = np.mean(clustering_coefficients) if clustering_coefficients else 0

        # Prepare problem-level statistics
        prob_stats = np.array([self.num_nodes, num_edges, total_weight, average_weight, max_weight, min_weight, density,
                               max_degree, min_degree, isolated_nodes, average_clustering_coefficient, weight_std_dev])

        # Variable-level statistics: degrees and clustering coefficients of each node
        var_stats = np.array([[degrees[i], self.graph[i][i], clustering_coefficients[i]] for i in range(self.num_nodes)])  # including self-loops and clustering coefficient

        return prob_stats, var_stats
    
    # def get_static_stats(self):
    #     """
    #     Computes and returns static statistics related to the max cut problem.

    #     Returns:
    #         tuple: A tuple containing two NumPy arrays:
    #             - prob_stats: Problem-level statistics.
    #             - var_stats: Variable-level statistics.
    #     """
    #     # Calculate max cut specific statistics
    #     node_degrees = [np.sum(self.graph[i]) for i in range(self.num_nodes)]
    #     edge_weights = [self.graph[i][j] for i in range(self.num_nodes) for j in range(i+1, self.num_nodes)]

    #     # Calculate node centrality measures
    #     betweenness_centrality = self.calculate_betweenness_centrality()
    #     closeness_centrality = self.calculate_closeness_centrality()

    #     # Calculate graph density
    #     num_edges = np.sum(self.graph) / 2
    #     max_edges = self.num_nodes * (self.num_nodes - 1) / 2
    #     graph_density = num_edges / max_edges

    #     # Create NumPy arrays for prob_stats and var_stats
    #     prob_stats = np.array([
    #         self.num_nodes,
    #         num_edges,
    #         graph_density,
    #         np.mean(edge_weights),
    #         np.std(edge_weights),
    #         # ... (Add other problem-level statistics)
    #     ])

    #     var_stats = np.zeros((self.num_nodes, 5))  # Assuming 5 variable-level statistics
    #     for i in range(self.num_nodes):
    #         var_stats[i] = np.array([
    #             node_degrees[i],
    #             betweenness_centrality[i],
    #             closeness_centrality[i],
    #             # ... (Add other variable-level statistics)
    #         ])

    #     return prob_stats, var_stats
