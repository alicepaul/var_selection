from pyscipopt import Model, quicksum
import numpy as np

class Problem:
    
    def __init__(self, x, gap_tol=1e-1):
        
        self.x = np.array(x)
        self.gap_tol = gap_tol
        self.num_sets, self.num_universe_items = self.x.shape
        
        # States
        self.prob_stats, self.var_stats = self.get_static_stats()

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
        model = Model("SetCoverRelaxed")
        set_vars = [model.addVar(vtype="C", name=f"x_{i}") for i in range(self.num_sets)]
        # Constraints
        for j in range(self.num_universe_items):
            model.addCons(quicksum(self.x[i, j] * set_vars[i] for i in range(self.num_sets)) >= 1)
        # Objective
        model.setObjective(quicksum(set_vars), "minimize")
        model.setIntParam("display/verblevel", 1)
        lp_vars = [var for var in model.getVars()]
        for i in node.zlb:
            model.chgVarUb(lp_vars[i], 0)
        for i in node.zub:
            model.chgVarLb(lp_vars[i], 1)
        model.setParam("presolving/maxrounds", 0)
        model.setParam("presolving/maxrestarts", 0)
        model.optimize()

        ## check if solution is infeasible
        if model.getStatus() == "infeasible":
            #print("Infeasible")
            node.primal_value = 100000
            node.dual_value = 100000
            return np.inf, np.inf
        primal_value = model.getPrimalbound()
        dual_value = model.getDualbound()

        # Update the attributes of the provided Node instance
        node.primal_value = primal_value
        node.dual_value = dual_value
        node.primal_beta = [model.getVal(var) for var in lp_vars]
        node.z = [model.getVal(var) for var in lp_vars]
        node.support = [i for i, var in enumerate(lp_vars) if model.getVal(var) > 0 and i not in node.zub]
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
        selected_sets = [i for i, val in enumerate(node.z) if val >= threshold]

        # Ensure all items are covered
        for j in range(num_universe_items):
            if not any(self.x[i, j] for i in selected_sets):
                # Find the set with the highest fractional value that covers this item
                sets_covering_item = [i for i in range(num_sets) if self.x[i, j] > 0]
                max_contrib_set = max(sets_covering_item, key=lambda i: node.z[i])
                selected_sets.append(max_contrib_set)

        selected_sets = list(set(selected_sets))  # Remove duplicates
        upper_bound = sum(set_costs[i] for i in selected_sets)
        # print("At upperBound: ", upper_bound, "selected sets: ", selected_sets)
        node.upper_bound = upper_bound
        node.upper_z = [1 if i in selected_sets else 0 for i in range(len(self.x[0]))]
        return upper_bound
    def get_static_stats(self):
        """
        Computes and returns static statistics related to the set cover problem.

        Returns:
            tuple: A tuple containing two NumPy arrays:
                - prob_stats: Problem-level statistics.
                - var_stats: Variable-level statistics.
        """

        # Calculate set cover specific statistics
        set_costs = [1 for _ in range(self.num_sets)]  # Replace with actual costs if needed
        set_coverage = [np.sum(self.x[i]) for i in range(self.num_sets)]
        item_coverage = [np.sum(self.x[:, j]) for j in range(self.num_universe_items)]

        # Calculate overlap (e.g., Jaccard index)
        overlap = np.zeros((self.num_sets, self.num_sets))
        for i in range(self.num_sets):
            for j in range(i + 1, self.num_sets):
                intersection = np.sum(np.logical_and(self.x[i], self.x[j]))
                union = np.sum(np.logical_or(self.x[i], self.x[j]))
                overlap[i, j] = intersection / union
                overlap[j, i] = overlap[i, j]

        # Create NumPy arrays for prob_stats and var_stats
        prob_stats = np.array([
            self.num_sets,
            self.num_universe_items,
            # ... (Add other problem-level statistics)
        ])

        var_stats = np.zeros((self.num_sets, 4))  # Assuming 6 variable-level statistics
        for i in range(self.num_sets):
            var_stats[i] = np.array([
                set_costs[i],
                set_coverage[i],
                item_coverage[i],  # Add item coverage for each set
                overlap[i].mean(),  # Add average overlap with other sets
                # ... (Add other variable-level statistics)
            ])

        return prob_stats, var_stats
    