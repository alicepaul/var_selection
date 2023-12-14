import numpy as np
from copy import deepcopy
from node import Node
from random import choice, choices
import math
from operator import attrgetter
from settings import MAX_ITERS
from scipy import optimize as sci_opt

def reverse_lookup(d, val):
  for key in d:
    if d[key] == val:
      return key


class tree():
    def __init__(self, problem):
        self.problem = problem
        self.tree_stats = None

        # Algorithm variables - also reset with set_root below
        self.active_nodes = dict()      # Current Nodes
        self.all_nodes = dict()         # All Nodes
        self.step_counter = 0           # Number branch steps taken
        self.node_counter = 0
        self.best_int = math.inf            # Best integer solution value
        self.candidate_sol = None           # Best integer solution betas
        self.lower_bound = None             # Minimum relaxation value of all nodes
        ## Check if needed
        self.initial_optimality_gap = None  # Initial optimality gap from root node
        self.optimality_gap = None          # Current optimality gap
        self.int_tol = problem.int_tol
        self.gap_tol = problem.gap_tol
        self.root = None

    def start_root(self):
        # Initializes the nodes with a root node
        # Use warm start for upper bound
        # Initialize root node
        root_node = Node(parent=None, node_key='root_node',zlb=[], zub=[], model = self.problem.model)
        self.active_nodes['root_node'] = root_node
        self.all_nodes['root_node'] = root_node
        ## TODO Modify according to new problem class
        self.problem.lower_solve(root_node)
        self.problem.upper_solve(root_node)

        # Update bounds and opt gap
        self.lower_bound = root_node.primal_value
        if (root_node.upper_bound < self.best_int):
            self.best_int = root_node.upper_bound
            self.candidate_sol = root_node.upper_z
        self.initial_optimality_gap = (self.best_int - self.lower_bound) / self.lower_bound
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
        """
        Returns tree statistics

        Returns:
            numpy.ndarray: Tree specific statistics.
                - number of steps taken
                - number of active nodes
                - number of candidate solution variables
                - lower bound
                - best integer solution
                - initial optimality gap
                - current optimality gpa
        """
        # Returns summary statisitics that describe the set of all active nodes
        tree_stats = np.array([self.step_counter,
                               len(self.active_nodes),
                               len(self.candidate_sol),
                               self.lower_bound,
                               self.best_int,
                               self.initial_optimality_gap,
                               self.optimality_gap])
                               
        return(tree_stats)

    def get_node_stats(self, node_key):
        """
        Returns node statistics

        Returns:
            numpy.ndarray: Node specific statistics.
                - length of zub
                - legth of zlb
                - node specific primal value
                - node depth level
                - legth of support
                - whether the node has the lower bound
                - whether the node has the upper bound
        """
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
        """
        Returns variable statistics

        Returns:
            numpy.ndarray: Node and Variable specific statistics.
                - primal beta
                - z value
                - upper z value 
        """
        node = self.all_nodes[node_key]

        # Case when node has no support (used for retrobranching)
        if len(node.support) == 0: 
            print('No Support found for Node during Retrobranching')
            # Values chosen to show lack of information for agent
            return np.array([0, -1, 0])
        
        # Returns summary stats for branching on j in a node
        index = [i for i in range(len(node.support)) if node.support[i] == j][0]
        var_stats = np.array([node.primal_beta[index],
                          node.z[index], node.upper_z[index]])
        
        return(var_stats)
    
    def get_state(self, node_key, j):
        '''
        Returns numpy array of 34 values containing problem and variable static states
        as well as tree, node, and variable current states.
        '''
        # Concatenates overall state for possible branch
        state = np.concatenate((self.problem.prob_stats,
                  self.problem.var_stats[j,:],
                  self.tree_stats,
                  self.get_node_stats(node_key),
                  self.get_var_stats(node_key, j)))
        return(state)

    def max_frac_branch(self):
        # Finds node with greatest fractional part
        best_node_key = None
        best_frac = 0
        best_index = None
        for node_key in self.active_nodes:
            node = self.active_nodes[node_key]
            z = node.z
            support = node.support
            diff = [min(1-z[i],z[i]) for i in range(len(support))]
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
                # beta_i = node.primal_beta[node.support.index(i)]
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
        ## Call lower solve from problem on current node
        self.problem.lower_solve(curr_node)
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
        self.active_nodes[node_name_1] = Node(parent=branch_node, node_key=node_name_1, zlb=new_zlb, zub=branch_node.zub, model = self.problem.model)
        self.node_counter += 1

        # Branch to beta_j to be 1
        node_name_2 = f'node_{self.node_counter}'
        self.active_nodes[node_name_2] = Node(parent=branch_node, node_key=node_name_2, zlb=branch_node.zlb, zub=new_zub, model = self.problem.model)
        self.node_counter += 1

        # Store New Nodes in all_nodes Dictionary
        self.all_nodes[node_name_1] = self.active_nodes[node_name_1]
        self.all_nodes[node_name_2] = self.active_nodes[node_name_2]

        # Store Child Nodes in Tree
        branch_node.assign_children(self.active_nodes[node_name_1], self.active_nodes[node_name_2])
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
            node_key, j = self.max_frac_branch()
            # Take a step
            fin_solving, old_gap, new_gap = self.step(node_key, j)

            # Update iterations and number positive
            if old_gap > new_gap:
                num_pos += 1
            iters += 1
        
        reward = -iters + 1

        return(iters, reward, len(self.candidate_sol), self.optimality_gap)

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

        
