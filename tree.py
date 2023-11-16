import numpy as np
from copy import deepcopy
from node import Node
from random import choice, choices
import math
from operator import attrgetter
from settings import MAX_ITERS
from scipy import optimize as sci_opt
from l0bnb.relaxation import cd_solve, l0gurobi, l0mosek

class Problem:
    """
    A class designed for a branch and bound algorithm to solve optimization problems, 
    containing methods for solving lower and upper bounds and calculating various statistics.

    Attributes:
        x (numpy.ndarray): Input matrix (features).
        y (numpy.ndarray): Output vector (target variable).
        l0 (float): Coefficient for the L0 norm regularization term.
        l2 (float): Coefficient for the L2 norm regularization term.
        m (float): A constant used in some calculations. Defaults to 1.5.
        int_tol (float): Tolerance for interpreting float as integer in optimization. Defaults to 1e-4.
        gap_tol (float): Tolerance for the gap in optimization process. Defaults to 1e-4.
        zlb (list): Lower bound for some optimization methods. Initialized as empty list.
        zub (list): Upper bound for some optimization methods. Initialized as empty list.
        xi_norm (numpy.ndarray): Stores the squared L2 norm of x.
    """
    def __init__(self, x, y, l0, l2, m = 1.5, int_tol=1e-4, gap_tol=1e-4):
        self.x = x
        self.xi_norm =  np.linalg.norm(self.x, axis=0) ** 2 ## for current usage
        self.y = y
        self.l0 = l0
        self.l2 = l2
        self.m = m 
        self.int_tol = int_tol
        self.gap_tol = gap_tol
        self.gs_xtr = None
        self.gs_xb = None
        self.r = None
        
    def lower_solve(self, node, solver='l1cd', rel_tol=1e-4, int_tol=1e-6,tree_upper_bound=None, mio_gap=None, cd_max_itr=100, kkt_max_itr=100):
        """
        Solves the lower bound problem for a given node in the branch and bound tree.

        Parameters:
            node (Node): The current node in the branch and bound tree.
            solver (str): The type of solver to use ('l1cd', 'gurobi', 'mosek', etc.). Defaults to 'l1cd'.
            rel_tol (float): Relative tolerance for the solver. Defaults to 1e-4.
            int_tol (float): Tolerance for interpreting float as integer in optimization. Defaults to 1e-6.
            tree_upper_bound (float, optional): Upper bound of the tree.
            mio_gap (float, optional): Gap for mixed integer optimization.
            cd_max_itr (int): Maximum iterations for coordinate descent. Defaults to 100.
            kkt_max_itr (int): Maximum iterations for KKT conditions. Defaults to 100.

        Returns:
            float: The primal value of the solution.
            float: The dual value of the solution.
        """
        if solver == 'l1cd':
            sol = cd_solve(x=self.x, y=self.y, l0=self.l0, l2=self.l2, m=self.m, zlb=node.zlb,
                           zub=node.zub, xi_norm=self.xi_norm, rel_tol=rel_tol,
                           warm_start=node.warm_start, r=self.r,
                           tree_upper_bound=tree_upper_bound, mio_gap=mio_gap,
                           gs_xtr=self.gs_xtr, gs_xb=self.gs_xb,
                           cd_max_itr=cd_max_itr, kkt_max_itr=kkt_max_itr)
            node.primal_value = sol.primal_value
            node.dual_value = sol.dual_value
            node.primal_beta = sol.primal_beta
            node.z = sol.z
            node.support = sol.support
            self.r = sol.r
            self.gs_xtr = sol.gs_xtr
            self.gs_xb = sol.gs_xb
        else:
            full_zlb = np.zeros(self.x.shape[1])
            full_zlb[node.zlb] = 1
            full_zub = np.ones(self.x.shape[1])
            full_zub[node.zub] = 0
            if solver == 'gurobi':
                primal_beta, z, self.primal_value, self.dual_value = \
                    l0gurobi(self.x, self.y, self.l0, self.l2, self.m, full_zlb, full_zub)
            elif solver == 'mosek':
                primal_beta, z, self.primal_value, self.dual_value = \
                    l0mosek(self.x, self.y, self.l0,self.l2, self.m, full_zlb, full_zub)
            else:
                raise ValueError(f'solver {solver} not supported')

            node.support = list(np.where(abs(primal_beta) > int_tol)[0])
            node.primal_beta = primal_beta[node.support]
            node.z = z[node.support]
        return node.primal_value, node.dual_value

    def upper_solve(self, node):
        """
        Solves the upper bound problem for a given node.

        Parameters:
            node (Node): The current node in the branch and bound tree.

        Returns:
            float: The calculated upper bound.
        """
        if len(node.support) != 0:
            x_support = self.x[:, node.support]
            x_ridge = np.sqrt(2 * self.l2) * np.identity(len(node.support))
            x_upper = np.concatenate((x_support, x_ridge), axis=0)
            y_upper = np.concatenate((self.y, np.zeros(len(node.support))), axis=0)
            # TODO: account for intercept later
            res = sci_opt.lsq_linear(x_upper, y_upper, (-self.m,  self.m))
            upper_bound = res.cost +  self.l0 * len(node.support)
            upper_z = res.x
        else:
            upper_bound = 0.5 * np.linalg.norm(self.y) ** 2
            upper_z = []
        
        node.upper_bound = upper_bound
        node.upper_z = upper_z
        return upper_bound
    
    def upper_bound_solve(self, support):
        """
        Computes the upper bound and corresponding beta values for a given node.

        Parameters:
            node (Node): The current node in the branch and bound tree.

        Returns:
            float: The computed upper bound.
            numpy.ndarray: The coefficients corresponding to this upper bound.
        """
        if len(support) != 0:
            x_support = self.x[:, support]
            x_ridge = np.sqrt(2 * self.l2) * np.identity(len(support))
            x_upper = np.concatenate((x_support, x_ridge), axis=0)
            y_upper = np.concatenate((self.y, np.zeros(len(support))), axis=0)
            # TODO: account for intercept later
            res = sci_opt.lsq_linear(x_upper, y_upper, (-self.m, self.m))
            upper_bound = res.cost + self.l0 * len(support)
            upper_z = res.x
        else:
            upper_bound = 0.5 * np.linalg.norm(self.y) ** 2
            upper_z = []
        return upper_bound, upper_z
    
    def get_static_stats(self):
        """
        Computes and returns static statistics related to the problem.

        Returns:
            numpy.ndarray: Problem-level statistics.
            numpy.ndarray: Variable-level statistics.
        """
        all_x_dot_y = np.matmul(self.x.T, self.y)
        cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)
        q = cov.shape[0]
        cov_flat = np.partition(cov.flatten(), kth=-q)[:-q]
        prob_stats = np.append(np.quantile(cov_flat, [0, 0.25, 0.5, 0.75, 1]), 
                               np.quantile(all_x_dot_y, [0, 0.25, 0.5, 0.75, 1]))
        p = self.x.shape[1]
        var_stats = np.zeros((p, 6))
        for i in range(p):
            x_i_cov = np.partition(cov[i, :], -1)[:-1]
            var_stats[i, 0:5] = np.quantile(x_i_cov, [0, 0.25, 0.5, 0.75, 1])
            var_stats[i, 5] = np.dot(self.x[:, i], self.y)

        return prob_stats, var_stats

def reverse_lookup(d, val):
  for key in d:
    if d[key] == val:
      return key


class tree():
    def __init__(self, problem):
        self.problem = problem
        self.prob_stats, self.var_stats = self.problem.get_static_stats()
        self.tree_stats = None

        # Algorithm variables - also reset with set_root below
        self.active_nodes = dict()      # Current Nodes
        self.all_nodes = dict()         # All Nodes
        self.step_counter = 0           # Number branch steps taken
        self.node_counter = 0
        self.best_int = math.inf        # Best integer solution value
        self.candidate_sol = None           # Best integer solution betas
        self.lower_bound = None         # Minimum relaxation value of all nodes
        self.initial_optimality_gap = None  # Initial optimality gap from root node
        self.optimality_gap = None          # Current optimality gap
        self.int_tol = problem.int_tol
        self.gap_tol = problem.gap_tol
        self.root = None

    def start_root(self, warm_start):
        # Initializes the nodes with a root node
        # Use warm start for upper bound
        if (warm_start is not None):
            support = np.nonzero(warm_start)[0]
            self.best_int_primal, self.best_int_beta = self.problem.upper_bound_solve(root_node, support)

        # Initialize root node
        root_node = Node(parent=None, node_key='root_node',zlb=[], zub=[])
        self.active_nodes['root_node'] = root_node
        self.all_nodes['root_node'] = root_node
        self.problem.lower_solve(root_node, solver='l1cd', rel_tol= 1e-4, mio_gap=1e-4,int_tol=self.int_tol)
        self.problem.upper_solve(root_node)

        # Update bounds and opt gap
        self.lower_bound = root_node.primal_value
        if (root_node.upper_bound < self.best_int):
            self.best_int = root_node.upper_bound
            self.candidate_sol = root_node.upper_z
        self.initial_optimality_gap = \
            (self.best_int - self.lower_bound) / self.lower_bound
        self.optimality_gap = self.initial_optimality_gap
        self.node_counter += 1
        self.tree_stats = self.get_tree_stats()

        # Start Tree
        self.root = self.active_nodes['root_node']

        # Return if done
        if self.int_sol(root_node) or (self.optimality_gap <= self.gap_tol):
            return(True)
        return(False)

    def get_info(self):
        # Summary info of current state
        info = {'step_count': self.step_counter,
                'node_count': self.node_counter,
                'best_primal_value': self.best_int,
                'candidate_sol': self.candidate_sol,
                'lower_bound': self.lower_bound,
                'init_opt_gap' : self.initial_optimality_gap,
                'opt_gap': self.optimality_gap}
        return(info)



    def get_tree_stats(self):
        # Returns summary statisitics that describe the set of all active nodes
        tree_stats = np.array([self.node_counter,
                               self.step_counter,
                               len(self.active_nodes),
                               self.lower_bound,
                               self.best_int,
                               self.initial_optimality_gap,
                               self.optimality_gap])
                               
        return(tree_stats)

    def get_node_stats(self, node_key):
        # Returns summary stats for a given node
        node = self.all_nodes[node_key]
        len_support = len(node.support) if node.support else 0
        has_lb = (self.lower_bound == node.primal_value)
        has_ub = (self.best_int == node.upper_bound)
        node_stats = np.array([len(node.zub),
                               len(node.zlb),
                               node.primal_value,
                               node.level,
                               len_support,
                               has_lb,
                               has_ub])
        return(node_stats)

    def get_var_stats(self, node_key, j):
        node = self.all_nodes[node_key]

        # Case when node has no support (used for retrobranching)
        if len(node.support) == 0: 
            print('No Support')
            return np.array([0,0])
        
        # Returns summary stats for branching on j in a node
        index = [i for i in range(len(node.support)) if node.support[i] == j][0]
        var_stats = np.array([node.primal_beta[index],
                          node.z[index]])
        return(var_stats)
    
    def get_state(self, node_key, j):
        # Concatenates overall state for possible branch
        state = np.concatenate((self.prob_stats,
                  self.var_stats[j,:],
                  self.tree_stats,
                  self.get_node_stats(node_key),
                  self.get_var_stats(node_key, j)))
        return(state)

    def get_frac_branchs(self):
        # Finds fractional part for all possible splits
        frac_dict = {}
        for node_key in self.active_nodes:
            node = self.active_nodes[node_key]
            for i in range(len(node.support)):
                frac_dict[(node_key, node.support[i])] = min([1-node.z[i],node.z[i]])
        return(frac_dict)

    def max_frac_branch(self):
        # Finds node with greatest fractional part
        best_node_key = None
        best_frac = 0
        best_index = None
        for node_key in self.active_nodes:
            node = self.active_nodes[node_key]
            z = node.z
            support = node.support
            diff = [min(1-z[i],z[i]-0) for i in range(len(support))]
            max_diff = max(diff)
            potential_j = [i for i in range(len(support)) if diff[i]== max_diff][0]
            if max_diff > best_frac:
                best_node_key = node_key
                best_frac = max_diff
                best_index = support[potential_j]
        return(best_node_key, best_index)

    def int_sol(self, node):
        # Check if within tolerance to an integer solution
        for i in node.support:
            if i not in node.zlb and i not in node.zub:
                #beta_i = node.primal_beta[node.support.index(i)]
                z_i = node.z[node.support.index(i)]
                residual = min(z_i, 1-z_i)
                if residual > self.int_tol:
                    return(False)
        return(True)

    def prune(self, node_keys):
        # Iterate through node keys to find those that aren't active
        for node_key in node_keys:
            if self.active_nodes[node_key].primal_value > self.best_int:
                del self.active_nodes[node_key]

    def update_lower_bound(self):
        # Update the best lower bound over all active nodes
        if len(self.active_nodes) == 0:
            self.lower_bound = self.best_int
        else:
            self.lower_bound = \
                             min(self.active_nodes.values(), \
                                 key=attrgetter('primal_value')).primal_value
            self.lower_bound_node_key = reverse_lookup(self.active_nodes, \
                                                       min(self.active_nodes.values(), \
                                                           key=attrgetter('primal_value')))
    def solve_node(self, node_key):
        # Solves a node with CD and updates upper bound 
        curr_node = self.active_nodes[node_key]
        self.problem.lower_solve(curr_node,solver='l1cd', rel_tol=1e-4, mio_gap=0,int_tol=self.int_tol)
        
        # Update upper bound by rounding soln
        curr_upper_bound = self.problem.upper_solve(curr_node)
        if curr_upper_bound < self.best_int:
            self.best_int = curr_upper_bound
            self.candidate_sol = curr_node.upper_z
            upper_bound_updated = True

        # Check if an integer solution
        if self.int_sol(curr_node):
            if curr_node.primal_value < self.best_int:
                self.best_int = curr_node.primal_value
                self.candidate_sol = curr_node.primal_beta
            del self.active_nodes[node_key]

    def step(self, branch_node_key, j):
        # Add State to Node in Tree
        branch_node = self.active_nodes[branch_node_key]
        branch_node.state = self.get_state(branch_node_key, j)

        # Branches for beta_j in node branch_node_key and returns True if done
        self.step_counter += 1

        # Create two child nodes
        branch_node = self.active_nodes[branch_node_key]
        new_zlb = branch_node.zlb+[j]
        new_zub = branch_node.zub+[j]
        
        # Branch to beta_j to be 0
        node_name_1 = f'node_{self.node_counter}'
        self.active_nodes[node_name_1] = Node(parent=branch_node, node_key=node_name_1, zlb=new_zlb, zub=branch_node.zub)
        self.node_counter += 1

        # Branch to beta_j to be 1
        node_name_2 = f'node_{self.node_counter}'
        self.active_nodes[node_name_2] = Node(parent=branch_node, node_key=node_name_2, zlb=branch_node.zlb, zub=new_zub)
        self.node_counter += 1

        # Store New Nodes in all_nodes Dictionary
        self.all_nodes[node_name_1] = self.active_nodes[node_name_1]
        self.all_nodes[node_name_2] = self.active_nodes[node_name_2]

        # Store Child Nodes in Tree
        branch_node.assign_children(self.active_nodes[node_name_1], \
                                   self.active_nodes[node_name_2])
        del self.active_nodes[branch_node_key] # Delete Parent from Active Nodes

        # Solve relaxations in new nodes
        past_upper = self.best_int
        self.solve_node(node_name_1)
        self.solve_node(node_name_2)

        # Check for node eliminations based on upper bound
        if (past_upper > self.best_int):
            self.prune(list(self.active_nodes.keys()))
            
        # Update lower_bound and lower_bound_node_key
        self.update_lower_bound()

        # Update optimality gap
        old_gap = self.optimality_gap
        self.optimality_gap = (self.best_int - self.lower_bound) / self.lower_bound

        # Update stats
        self.tree_stats = self.get_tree_stats()

        # Return True if solved or within tolerance and False otherwise
        if (len(self.active_nodes) == 0) or (self.optimality_gap <= self.gap_tol):
            return(True, old_gap, self.optimality_gap)
        return(False, old_gap, self.optimality_gap)

    def branch_and_bound(self, branch="max"):
        self.start_root(None)

        fin_solving = False
        iters = 0
        num_pos = 0
        while (not fin_solving) and (iters < MAX_ITERS):
            # Find node based on branching strategy
            if branch == "max":
                node_key, j = self.max_frac_branch()
            elif branch == "sample":
                frac_dict = self.get_frac_branchs()
                vals = list(frac_dict.values())
                node_key, j = choices(list(frac_dict.keys()), weights=vals)[0]
            elif branch == "strong_branch":
                frac_dict = self.get_frac_branchs()
                vals = list(frac_dict.values())
                pot_branches = choices(list(frac_dict.keys()), k=5, weights=vals)
                best_val = 0
                node_key, j = pot_branches[0]
                for node_key_i, j_i in pot_branches:
                    T_prime = deepcopy(self)
                    fin_solving, old_gap, new_gap = T_prime.step(node_key_i, j_i)
                    if old_gap - new_gap > best_val:
                        best_val = old_gap - new_gap
                        node_key, j = node_key_i, j_i
            else:
                node_key = choice(list(self.active_nodes))
                j = choice(self.active_nodes[node_key].support)

            # Take a step
            fin_solving, old_gap, new_gap = self.step(node_key, j)

            # Update iterations and number positive
            if old_gap > new_gap:
                num_pos += 1
            iters += 1

        # Complete Tree (Get's states for leaf nodes)
        for node in self.all_nodes.values():
            if node.state is None:
                z = node.z
                support = node.support
                diff = [min(1 - z[i], z[i] - 0) for i in range(len(support))]
                j = support[np.argmax(diff)]
                node.state = self.get_state(node.node_key, j)

        
        reward = -iters + 1

        return(iters, reward, len(self.candidate_sol))

    # The get_state_pairs function can remain unchanged if it's still required.

    def get_state_pairs(self, node):
        '''
        Recursively collect tree edges, parent and child states pairs

        input:
            node: node of tree
        return:
            list of (previous state, state, reward) tuples
        '''
        pairs = []

        if not node:
            return pairs

        # Check left child
        if node.left:
            if node.left.is_leaf:
                # If child is leaf reward is 0
                pairs.append((node.state, node.left.state, 0))
            else:
                pairs.append((node.state, node.left.state, -1))
            pairs.extend(self.get_state_pairs(node.left))

        # Check right child
        if node.right:
            if node.right.is_leaf:
                # If child is leaf reward is -1
                pairs.append((node.state, node.right.state, 0))
            else:
                pairs.append((node.state, node.right.state, -1))
            pairs.extend(self.get_state_pairs(node.right))

        return pairs

    
