class Node:
    def __init__(self, parent, node_key, zlb, zub):
        """
        Initialize a Node for the set cover problem.

        Parameters:
            parent: Node or None
                The parent Node.
            node_key: str
                Name associated with Node, used for Node Lookup.
            zlb: list
                Sets that must be included.
            zub: list
                Sets that are yet to be decided.
        """
        self.parent_dual = parent.dual_value if parent else None
        self.parent_primal = parent.primal_value if parent else None
        self.level = parent.level + 1 if parent else 0
        self.zlb = zlb
        self.zub = zub

        self.upper_bound = None
        self.primal_value = None
        self.dual_value = None
        self.lp_solution = None  # Stores the LP relaxation solution within each node
        self.is_feasible = None  # Indicates if the node represents a feasible solution

        self.node_key = node_key
        self.parent_key = parent.node_key if parent else None
        self.is_leaf = True 
        self.left = None
        self.right = None
        self.state = None
        
        # Additional attributes for heuristic-based features
        self.aggregated_density_included = 0   # For included sets
        self.aggregated_overlap_included = 0   # For included sets
        self.aggregated_density_undecided = 0  # For undecided sets
        self.aggregated_overlap_undecided = 0  # For undecided sets

    def get_info(self):
        return f'Node Key: {self.node_key}, Level: {self.level}, ' \
               f'Primal Value: {self.primal_value}, Dual Value: {self.dual_value}, ' \
               f'Upper Bound: {self.upper_bound}, Is Leaf: {self.is_leaf}, State: {self.state}'

    def assign_children(self, left_child=None, right_child=None):
        """
        Assigns children nodes to the current node and updates leaf status.

        Parameters:
            left_child: Node
                Node associated with Left Child.
            right_child: Node
                Node associated with Right Child.
        """
        if left_child:
            self.left = left_child
            self.left.parent_key = self.node_key
            self.is_leaf = False

        if right_child:
            self.right = right_child
            self.right.parent_key = self.node_key
            self.is_leaf = False

    def update_lp_solution(self, lp_solution):
        """
        Updates the LP solution for the node.

        Parameters:
            lp_solution: list
                The LP solution values for this node.
        """
        self.lp_solution = lp_solution
    
    def update_heuristic_features(self, set_density, coverage_overlap):
        """
        Updates the heuristic features of the node.

        Parameters:
            set_density: dict
                Density of each set.
            coverage_overlap: dict
                Overlap in coverage between different sets.
        """
        # Calculate aggregated metrics for included sets (zlb)
        self.aggregated_density_included = sum(set_density[i] for i in self.zlb)
        self.aggregated_overlap_included = sum(coverage_overlap[i] for i in self.zlb)

        # Calculate aggregated metrics for undecided sets (zub)
        self.aggregated_density_undecided = sum(set_density[i] for i in self.zub)
        self.aggregated_overlap_undecided = sum(coverage_overlap[i] for i in self.zub)

    
        
    def update_heuristic_features(self, set_density, coverage_overlap):
        """
        Updates the heuristic features of the node.

        Parameters:
            set_density: dict
                Density of each set.
            coverage_overlap: dict
                Overlap in coverage between different sets.
        """
        self.set_density = set_density
        self.coverage_overlap = coverage_overlap
        
    def check_feasibility(self):
        """
        Checks and updates the feasibility of the node.
        """
        # Implement feasibility check logic here
        # Example: self.is_feasible = True if all values in lp_solution are integral else False
        self.is_feasible = all(value.is_integer() for value in self.lp_solution)

    def __str__(self):
        return f'Node Key: {self.node_key}, Level: {self.level}, Primal Value: {self.primal_value}, ' \
               f'Agg Density Inc: {self.aggregated_density_included}, Agg Overlap Inc: {self.aggregated_overlap_included}, ' \
               f'Agg Density Und: {self.aggregated_density_undecided}, Agg Overlap Und: {self.aggregated_overlap_undecided}, ' \
               f'Feasible: {self.is_feasible}'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.level < other.level or (self.level == other.level and self.primal_value > other.primal_value)
