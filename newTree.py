import numpy as np
from copy import deepcopy
from newNode import Node
from random import choice, choices
import math
from operator import attrgetter
from settings import MAX_ITERS
from scipy import optimize as sci_opt
from l0bnb.relaxation import cd_solve, l0gurobi, l0mosek

## implement max cut
class Problem:
    def __init__(self, x, y, l0, l2, m = 1.5, int_tol=1e-4, gap_tol=1e-4):
        self.x = x
        self.xi_norm =  np.linalg.norm(self.x, axis=0) ** 2 ## for current usage
        self.y = y
        self.l0 = l0
        self.l2 = l2
        self.m = m 
        self.int_tol = int_tol
        self.gap_tol = gap_tol
        self.zlb = list
        self.zub = list
        
    def lower_solve(self, node, solver='l1cd', rel_tol=1e-4, int_tol=1e-6,
                          tree_upper_bound=None, mio_gap=None, cd_max_itr=100,
                          kkt_max_itr=100):
        if solver == 'l1cd':
            sol = cd_solve(x=self.x, y=self.y, l0=self.l0, l2=self.l2, m=self.m, zlb=self.zlb,
                           zub=self.zub, xi_norm=self.xi_norm, rel_tol=rel_tol,
                           warm_start=node.warm_start, r=node.r,
                           tree_upper_bound=tree_upper_bound, mio_gap=mio_gap,
                           gs_xtr=node.gs_xtr, gs_xb=node.gs_xb,
                           cd_max_itr=cd_max_itr, kkt_max_itr=kkt_max_itr)
            node.primal_value = sol.primal_value
            node.dual_value = sol.dual_value
            node.primal_beta = sol.primal_beta
            node.z = sol.z
            node.support = sol.support
            node.r = sol.r
            node.gs_xtr = sol.gs_xtr
            node.gs_xb = sol.gs_xb
        else:
            full_zlb = np.zeros(self.x.shape[1])
            full_zlb[self.zlb] = 1
            full_zub = np.ones(self.x.shape[1])
            full_zub[self.zub] = 0
            if solver == 'gurobi':
                primal_beta, z, self.primal_value, self.dual_value = \
                    l0gurobi(self.x, self.y, self.l0, self.l2, self.m, full_zlb, full_zub)
            elif solver == 'mosek':
                primal_beta, z, self.primal_value, self.dual_value = \
                    l0mosek(self.x, self.y, self.l0,self.l2, self.m, full_zlb, full_zub)
            else:
                raise ValueError(f'solver {solver} not supported')

            node.support = list(np.where(abs(primal_beta) > int_tol)[0])
            node.primal_beta = primal_beta[self.support]
            node.z = z[node.support]
        return node.primal_value, node.dual_value

    def upper_solve(self, node):
        if len(node.support) != 0:
            x_support = self.x[:, node.support]
            x_ridge = np.sqrt(2 * self.l2) * np.identity(len(node.support))
            x_upper = np.concatenate((x_support, x_ridge), axis=0)
            y_upper = np.concatenate((self.y, np.zeros(len(node.support))), axis=0)
            # TODO: account for intercept later
            res = sci_opt.lsq_linear(x_upper, y_upper, (-self.m,  self.m))
            upper_bound = res.cost +  self.l0 * len(node.support)
            upper_beta = res.x
        else:
            upper_bound = 0.5 * np.linalg.norm(self.y) ** 2
            upper_beta = []
        
        node.upper_bound = upper_bound
        node.upper_beta = upper_beta
        return upper_bound
    
    def upper_bound_solve(self, node):
        if len(node.support) != 0:
            x_support = self.x[:, node.support]
            x_ridge = np.sqrt(2 * self.l2) * np.identity(len(node.support))
            x_upper = np.concatenate((x_support, x_ridge), axis=0)
            y_upper = np.concatenate((self.y, np.zeros(len(node.support))), axis=0)
            # TODO: account for intercept later
            res = sci_opt.lsq_linear(x_upper, y_upper, (-self.m, self.m))
            upper_bound = res.cost + self.l0 * len(node.support)
            upper_beta = res.x
        else:
            upper_bound = 0.5 * np.linalg.norm(self.y) ** 2
            upper_beta = []
        return upper_bound, upper_beta
    


def reverse_lookup(d, val):
  for key in d:
    if d[key] == val:
      return key


class tree():
    def __init__(self, problem2):
        ## X_i norm is needed for every single node - yes, while calling upper and lower solve
        ## Does X_i norm change? - No
        ## move x_i norm to problem - line below
        ## xi_norm =  np.linalg.norm(self.x, axis=0) ** 2
        # Initialize a branch and bound tree for a given problem
        # Stats for the state
        self.problem = problem2
        self.prob_stats, self.var_stats = self.get_static_stats()
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
        # self.m = 1.5                        # Bound on betas

        # Tree Structure 
        self.root = None

    def start_root(self, warm_start):
        # Initializes the nodes with a root node

        # Use warm start for upper bound
        if (warm_start is not None):
            support = np.nonzero(warm_start)[0]
            self.best_int_primal, self.best_int_beta = \
                                  upper_bound_solve(root_node)

        # Initialize root node
        
        root_node = Node(parent=None, node_key='root_node')
        self.active_nodes['root_node'] = root_node
        self.all_nodes['root_node'] = root_node
        self.problem.lower_solve(root_node)
        self.problem.upper_solve(root_node)

        # Update bounds and opt gap
        self.lower_bound = root_node.primal_value
        if (root_node.upper_bound < self.best_int):
            self.best_int = root_node.upper_bound
            self.candidate_sol = root_node.upper_beta
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

    def get_static_stats(self):
        # Returns static stats plus a matrix of static stats for each variable
        all_x_dot_y = np.matmul(self.problem.x.T, self.problem.y)
        cov = self.problem.x.shape[0] * np.cov(self.problem.x, rowvar=False, bias=True)

        # get quantiles from cov
        q = cov.shape[0]
        cov_flat = np.partition(cov.flatten(), kth=-q)[:-q]
        prob_stats = np.append(np.quantile(cov_flat, [0, 0.25, 0.5, 0.75, 1]), \
            np.quantile(all_x_dot_y,[0,0.25,0.5,0.75,1]))

        # for each node store dot product and cov
        p = self.problem.x.shape[1]
        var_stats = np.zeros((p,6))
        for i in range(p):
            x_i_cov = np.partition(cov[i,:], -1)[:-1]
            var_stats[i,0:5] = np.quantile(x_i_cov,[0,0.25,0.5,0.75,1])
            var_stats[i,5] = np.dot(self.problem.x[:,i], self.problem.y)

        return(prob_stats, var_stats)

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
            if i not in self.problem.zlb and i not in self.problem.zub:
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
        self.problem.lower_solve(curr_node)
        
        # Update upper bound by rounding soln
        curr_upper_bound = self.problem.upper_solve(curr_node)
        if curr_upper_bound < self.best_int:
            self.best_int = curr_upper_bound
            self.candidate_sol = curr_node.upper_beta
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
        new_zlb =self.problem.zlb+[j]
        new_zub = self.problem.zub+[j]
        
        # Branch to beta_j to be 0
        node_name_1 = f'node_{self.node_counter}'
        self.active_nodes[node_name_1] = Node(parent=branch_node, node_key=node_name_1)
        self.node_counter += 1

        # Branch to beta_j to be 1
        node_name_2 = f'node_{self.node_counter}'
        self.active_nodes[node_name_2] = Node(parent=branch_node, node_key=node_name_2)
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

        return iters, num_pos, self

# The get_state_pairs function can remain unchanged if it's still required.

def get_state_pairs(node):
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
        pairs.extend(get_state_pairs(node.left))

    # Check right child
    if node.right:
        if node.right.is_leaf:
            # If child is leaf reward is -1
            pairs.append((node.state, node.right.state, 0))
        else:
            pairs.append((node.state, node.right.state, -1))
        pairs.extend(get_state_pairs(node.right))

    return pairs

    
