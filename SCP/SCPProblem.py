from pyscipopt import Model, quicksum

class Problem:
    """
    A class designed to solve the set cover problem using the branch and bound algorithm.
    
    Attributes:
        sets (list of lists): Collection of sets, each containing elements.
        universe (set): The universe of elements that need to be covered.
    """

    def __init__(self, sets, universe):
        self.sets = sets
        self.universe = universe

    def lower_solve(self):
        """
        Solves the relaxed set cover problem (linear programming relaxation).

        Returns:
            model (Model): The solved SCIP model for the relaxed problem.
        """
        model = Model("SetCoverRelaxed")
        x = {}

        # Variables (fractional allowed for relaxation)
        for j in range(len(self.sets)):
            x[j] = model.addVar(vtype="C", name="x_%s" % j)

        # Constraints
        for i in self.universe:
            model.addCons(quicksum(x[j] for j, s in enumerate(self.sets) if i in s) >= 1)

        # Objective
        model.setObjective(quicksum(x[j] for j in range(len(self.sets))), "minimize")

        model.optimize()
        return model

    def upper_solve(self):
        """
        Solves the set cover problem (integer version).

        Returns:
            model (Model): The solved SCIP model for the set cover problem.
        """
        model = Model("SetCover")

        # Variables (binary for set cover)
        x = {j: model.addVar(vtype="B", name="x_%s" % j) for j in range(len(self.sets))}

        # Constraints
        for i in self.universe:
            model.addCons(quicksum(x[j] for j, s in enumerate(self.sets) if i in s) >= 1)

        # Objective
        model.setObjective(quicksum(x[j] for j in range(len(self.sets))), "minimize")

        model.optimize()
        return model

# Example usage
sets = [{1, 2, 3}, {2, 4}, {3, 4}, {4, 5}]
universe = set().union(*sets)
problem = Problem(sets, universe)

relaxed_model = problem.lower_solve()
integer_model = problem.upper_solve()

# Get results
print("Relaxed Solution: ", relaxed_model.getObjVal())
print("Integer Solution: ", integer_model.getObjVal())
