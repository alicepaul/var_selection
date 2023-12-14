from pyscipopt import Model, quicksum
import numpy as np
import random

class Problem:
    
    def __init__(self, x):
        self.x = x
        # Setup the model
        self.model = Model("SetCoverRelaxed")
        num_sets, num_universe_items = self.x.shape
        set_vars = [self.model.addVar(vtype="C", name=f"x_{i}") for i in range(num_sets)]
        # Constraints
        for j in range(num_universe_items):
            self.model.addCons(quicksum(self.x[i, j] * set_vars[i] for i in range(num_sets)) >= 1)
        # Objective
        self.model.setObjective(quicksum(set_vars), "minimize")

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
        
        # Adding branching decisions from node state (zlb and zub)
        lp_vars = [var for var in node.model.getVars()]
        for i in node.zlb:
            node.model.chgVarUb(lp_vars[i], 0)
        for i in node.zub:
            node.model.chgVarLb(lp_vars[i], 1)
        
        node.model.optimize()

        primal_value = node.model.getPrimalbound()
        dual_value = node.model.getDualbound()

        # Update the attributes of the provided Node instance
        node.primal_value = primal_value
        node.dual_value = dual_value
        node.lp_solution = [node.model.getVal(var) for var in lp_vars]

        return node.primal_value, node.dual_value
    
    def upper_solve(self, node):
        """
        Solves the set cover problem using LP rounding to provide an upper bound.

        Parameters:
            node (Node): The current node in the branch and bound tree.

        Returns:
            float: The cost of the selected sets (upper bound).
            list: Indices of selected sets.
        """
        num_sets, num_universe_items = self.x.shape
        set_costs = [1 for _ in range(len(self.x))]  # Assuming each set has a cost of 1
        threshold = 0.5
        selected_sets = [i for i, val in enumerate(node.lp_solution) if val >= threshold]

        # Ensure all items are covered
        for j in range(num_universe_items):
            if not any(self.x[i, j] for i in selected_sets):
                # Find the set with the highest fractional value that covers this item
                sets_covering_item = [i for i in range(num_sets) if self.x[i, j] > 0]
                max_contrib_set = max(sets_covering_item, key=lambda i: node.lp_solution[i])
                selected_sets.append(max_contrib_set)

        selected_sets = list(set(selected_sets))  # Remove duplicates
        upper_bound = sum(set_costs[i] for i in selected_sets)
        upper_z = [1 if i in selected_sets else 0 for i in range(len(self.x[0]))]
        return upper_bound