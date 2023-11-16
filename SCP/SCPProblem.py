from pyscipopt import Model, quicksum
import numpy as np
import random

class Problem:
    """
    A class designed to solve the set cover problem using the branch and bound algorithm.
    
    Attributes:
        x (numpy.ndarray): Binary matrix representing the sets.
        universe (set): The universe of elements that need to be covered.
    """

    def __init__(self, x, universe):
        self.x = x
        self.universe = universe

    def lower_solve(self):
        """
        Solves the relaxed set cover problem (linear programming relaxation).

        Returns:
            model (Model): The solved SCIP model for the relaxed problem.
        """
        model = Model("SetCoverRelaxed")
        num_sets, num_universe_items = self.x.shape
        set_vars = [model.addVar(vtype="C", name=f"x_{i}") for i in range(num_sets)]

        # Constraints
        for j in range(num_universe_items):
            model.addCons(quicksum(self.x[i, j] * set_vars[i] for i in range(num_sets)) >= 1)

        # Objective
        model.setObjective(quicksum(set_vars), "minimize")

        model.optimize()
        return model

    def upper_solve(self):
        """
        Solves the set cover problem (integer version).

        Returns:
            model (Model): The solved SCIP model for the set cover problem.
        """
        model = Model("SetCover")
        num_sets, num_universe_items = self.x.shape
        set_vars = [model.addVar(vtype="B", name=f"x_{i}") for i in range(num_sets)]

        # Constraints
        for j in range(num_universe_items):
            model.addCons(quicksum(self.x[i, j] * set_vars[i] for i in range(num_sets)) >= 1)

        # Objective
        model.setObjective(quicksum(set_vars), "minimize")

        model.optimize()
        return model

    def upper_solve_lp_rounding(self):
        """
        Solves the set cover problem using LP rounding.

        Returns:
            list: Indices of selected sets.
        """
        # First, solve the relaxed problem
        relaxed_model = self.lower_solve()

        # Get the LP solution
        num_sets = len(self.sets)
        lp_solution = [relaxed_model.getVal(relaxed_model.getVarByName(f"x_{i}")) for i in range(num_sets)]

        # Rounding: Select sets where the LP solution value is greater than a threshold
        threshold = 0.5
        selected_sets = [i for i, val in enumerate(lp_solution) if val >= threshold]

        # Ensure all items are covered, add missing coverage
        for j in range(len(self.universe)):
            if not any(self.x[i, j] for i in selected_sets):
                # Find the set that contributes most to covering this item
                # based on the LP solution and add it to the selected sets
                max_contrib_set = max(range(num_sets), key=lambda i: self.x[i, j] * lp_solution[i])
                selected_sets.append(max_contrib_set)

        return list(set(selected_sets))  # Remove duplicates if any

def generate_scp_dataset(num_universe_items, num_sets, min_set_size, max_set_size):
    """
    Generate a dataset for the Set Covering Problem.

    :param num_universe_items: Number of items in the universe.
    :param num_sets: Number of sets to generate.
    :param min_set_size: Minimum number of items in each set.
    :param max_set_size: Maximum number of items in each set.
    :return: A tuple (universe, x) where universe is a set of all items,
             and x is a binary matrix representing the sets.
    """
    # Generate the universe of items
    universe = set(range(num_universe_items))

    # Initialize the binary matrix for sets
    x = np.zeros((num_sets, num_universe_items), dtype=int)

    # Generate the sets and update the binary matrix
    for i in range(num_sets):
        set_size = random.randint(min_set_size, max_set_size)
        set_items = random.sample(list(universe), set_size)  # Convert the set to a list here
        for item in set_items:
            x[i, item] = 1

    # Ensure every item is covered at least once
    uncovered_items = universe.copy()
    for item in universe:
        if not np.any(x[:, item]):
            random_set = random.randint(0, num_sets - 1)
            x[random_set, item] = 1
            uncovered_items.discard(item)

    # If there are still uncovered items, add them to random sets
    while uncovered_items:
        item = uncovered_items.pop()
        random_set = random.randint(0, num_sets - 1)
        x[random_set, item] = 1

    return universe, x

# # Example usage
# problem = Problem(x, universe)

# # Solve using LP rounding
# selected_sets_lp_rounding = problem.upper_solve_lp_rounding()
# print("Sets selected by LP Rounding:", selected_sets_lp_rounding)




num_universe_items = 50
num_sets = 20
min_set_size = 3
max_set_size = 10

universe, x = generate_scp_dataset(num_universe_items, num_sets, min_set_size, max_set_size)

# Create an instance of the Problem class with the generated data
problem = Problem(x, universe)

relaxed_model = problem.lower_solve()
integer_model = problem.upper_solve()

# Get results
print("Relaxed Solution: ", relaxed_model.getObjVal())
print("Integer Solution: ", integer_model.getObjVal())
