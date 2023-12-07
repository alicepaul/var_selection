from pyscipopt import Model, quicksum
import numpy as np
import random

class Problem:
    
    def __init__(self, x, universe):
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
    
    def max_frac_branch(self, lp_solution):
        """
        Finds the decision variable with the maximum fractional part in the LP solution.

        Parameters:
            lp_solution (list of float): The LP solution of the decision variables.

        Returns:
            int: The index of the decision variable with the maximum fractional part.
        """
        max_frac = -1
        max_frac_index = -1
        for i, val in enumerate(lp_solution):
            frac_part = abs(val - round(val))
            if frac_part > max_frac:
                max_frac = frac_part
                max_frac_index = i
        return max_frac_index
    
    def upper_solve_lp_rounding(self, node):
        """
        Method to solve the set cover problem using LP rounding and branching.

        Parameters:
            node (Node): The current node in the branch and bound tree.

        Returns:
            list: Indices of selected sets.
        """
        lp_solution = node.lp_solution
        max_frac_var = self.max_frac_branch(lp_solution)
        
        # Create two new nodes for branching
        node_with_var = node(node.zlb + [max_frac_var], node.zub)
        node_without_var = node(node.zlb, node.zub + [max_frac_var])

        # Solve both subproblems using lower_solve
        # this will move to step
        self.lower_solve(node_with_var)
        self.lower_solve(node_without_var)

        return node_with_var.lp_solution if node_with_var.primal_value < node_without_var.primal_value else node_without_var.lp_solution

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

