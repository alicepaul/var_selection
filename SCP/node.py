import numpy as np
import random 

class Node:
    def __init__(self, var_size, node_key):
        """
        Initialize a Node

        Parameters
        ----------
        var_size: Int
            Number of variables to select from in support
        
        node_key: Str
            Name associated with Node, used for Node Lookup

        """
        # Initialize z with all zeros
        self.z = [0] * var_size

        # Set one random variable to a value between 1 and 5
        random_index = random.randint(0, var_size - 1)
        self.z[random_index] = random.randint(1, 5)

        # Each node will store it's children's nodes and current state
        self.node_key = node_key
        self.parent_key = None
        self.is_leaf = True
        self.left = None
        self.right = None
        self.state = None
    
    def assign_children(self, left_child=None, right_child=None):
        '''
        Function assigns children nodes to parent and 
        parent is set to no longer leaf

        inputs:
            left_child: Node associated with Left Child
            right_child: Node associated with Right Child
        '''
        if left_child is not None:
            self.left = left_child
            self.left.parent_key = self.node_key
            self.is_leaf = False
        
        if right_child is not None:
            self.right = right_child
            self.right.parent_key = self.node_key
            self.is_leaf = False
from pyscipopt import Model, quicksum
import numpy as np
import random

class Problem:
    def __init__(self, x, universe):
        ## get rid of universe
        self.x = x
        self.universe = universe
        self.coverage_overlap = self.calculate_coverage_overlap()
        # self.set_density = self.calculate_set_density()

    def lower_solve(self, node):
        """
        Solves the relaxed set cover problem (linear programming relaxation).

        Parameters:
            node (Node): The current node in the branch and bound tree.

        Returns:
            tuple: 
                - primal_value (float): The primal value of the solution.
                - dual_value (float): The dual value of the solution.
        """
        model = Model("SetCoverRelaxed")
        num_sets, num_universe_items = self.x.shape
        set_vars = [model.addVar(vtype="C", name=f"x_{i}") for i in range(num_sets)]

        # Constraints
        for j in range(num_universe_items):
            model.addCons(quicksum(self.x[i, j] * set_vars[i] for i in range(num_sets)) >= 1)

        # Objective
        model.setObjective(quicksum(set_vars), "minimize")
        
        # Adding branching decisions from node state (zlb and zub)
        for i in node.zlb:
            model.chgVarLb(set_vars[i], 1)
        for i in node.zub:
            model.chgVarUb(set_vars[i], 0)
        
        model.optimize()

        primal_value = model.getPrimalbound()
        dual_value = model.getDualbound()

        # Update the attributes of the provided Node instance
        node.primal_value = primal_value
        node.dual_value = dual_value
        node.lp_solution = [model.getVal(var) for var in set_vars]

        return node.primal_value, node.dual_value
    
    def upper_solve_lp_rounding(self, node):
        """
        Solves the set cover problem using LP rounding.

        Returns:
            list: Indices of selected sets.
        """
        # The LP solution is stored in the node
        lp_solution = node.lp_solution
        threshold = 0.5
        ## 
        selected_sets = [i for i, val in enumerate(lp_solution) if val >= threshold]
        num_sets, num_universe_items = self.x.shape
        
        # Ensure all items are covered, add missing coverage
        for j in range(len(self.universe)):
            if not any(self.x[i, j] for i in selected_sets):
                # Find the set that contributes most to covering this item based on the LP solution and add it to the selected sets
                max_contrib_set = max(range(num_sets), key=lambda i: self.x[i, j] * lp_solution[i])
                selected_sets.append(max_contrib_set)

        return list(set(selected_sets))  

    def calculate_coverage_overlap(self):
        """
        Calculates the coverage overlap for each set in the set cover problem.

        Coverage overlap measures the degree of overlap in coverage between different sets.
        It is calculated as the number of elements that are also covered by other sets.

        Returns:
            dict: A dictionary where keys are set indices and values are their coverage overlap counts.
        """
        coverage_overlap = {}
        num_sets, num_elements = self.x.shape

        for i in range(num_sets):
            overlap_count = 0
            for j in range(num_elements):
                # Check if set 'i' covers element 'j'
                if self.x[i, j] == 1:
                    # Count how many other sets cover this element.
                    # Subtract 1 to exclude the current set 'i' from the count.
                    overlap_count += np.sum(self.x[:, j]) - 1
            # Store the total overlap count for set 'i'
            coverage_overlap[i] = overlap_count

        return coverage_overlap

    # def calculate_set_density(self):
    #     """
    #     Calculates the density for each set in the set cover problem.

    #     Density is defined as the number of elements a set covers relative to its size (the number of elements in it).
    #     It is a measure of how 'valuable' a set is in terms of coverage.

    #     Returns:
    #         dict: A dictionary where keys are set indices and values are their density.
    #     """
    #     set_density = {}
    #     num_sets, num_elements = self.x.shape

    #     for i in range(num_sets):
    #         set_size = np.sum(self.x[i, :])
    #         # Calculate density as the ratio of elements covered to the set size
    #         if set_size > 0:
    #             set_density[i] = np.sum(self.x[i, :]) / set_size
    #         else:
    #             set_density[i] = 0

    #     return set_density

def max_frac_branch(problem, node):
    """
    Finds the variable with the maximum fractional value for branching.

    Parameters:
        problem (Problem): The set cover problem instance.
        node (Node): The current node in the branch and bound tree.

    Returns:
        int: The index of the variable to branch on.
    """
    # Get the LP solution from the current node
    lp_solution = node.lp_solution

    # Find the variable index with the maximum fractional part
    max_frac_index = -1
    max_frac_value = -1

    for i, val in enumerate(lp_solution):
        # Calculate the fractional part of the value
        fractional_part = val - int(val)
        if fractional_part > max_frac_value:
            max_frac_index = i
            max_frac_value = fractional_part

    return max_frac_index
