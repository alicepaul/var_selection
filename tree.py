# Toby Dekara and Alice Paul
# Created July 14, 2022
# Tree structure for branch and bound process

#  Tree structure adapted from 'l0bnb', 
# Copyright (c) 2020 [Hussein Hazimeh, and Rahul Mazumder, Ali Saab]

import numpy as np
from copy import deepcopy
from l0bnb.node import Node, upper_bound_solve
from random import choice, choices
import math
from operator import attrgetter
from settings import MAX_ITERS

def reverse_lookup(d, val):
  for key in d:
    if d[key] == val:
      return key


class tree():
    def __init__(self, x, y, l0, l2, int_tol=1e-4, gap_tol=1e-4):
        # Initialize a branch and bound tree for a given problem
                
        # Problem definition variables
        self.x = x
        self.y = y
        self.L0 = l0
        self.L2 = l2
        self.int_tol = int_tol
        self.gap_tol = gap_tol

        # Stats for the state
        self.prob_stats, self.var_stats = self.get_static_stats()
        self.tree_stats = None

        # Algorithm variables - also reset with set_root below
        self.active_nodes = dict()      # Dictionary current nodes
        self.step_counter = 0           # Number branch steps taken
        self.node_counter = 0
        self.best_int = math.inf        # Best integer solution value
        self.best_beta = None           # Best integer solution betas
        self.lower_bound = None         # Minimum relaxation value of all nodes
        self.initial_optimality_gap = None  # Initial optimality gap from root node
        self.optimality_gap = None          # Current optimality gap
        self.m = 1.5                          # Bound on betas

    def start_root(self, warm_start):
        # Initializes the nodes with a root node

        # Use warm start for upper bound
        if (warm_start is not None):
            support = np.nonzero(warm_start)[0]
            self.best_int_primal, self.best_int_beta = \
                                  upper_bound_solve(self.x, self.y, self.L0,
                                                    self.L2, self.m, support)

        # Initialize root node
        xi_norm =  np.linalg.norm(self.x, axis=0) ** 2
        root_node = Node(parent=None, zlb=[], zub=[], x=self.x, y=self.y, \
                         xi_norm=xi_norm)
        self.active_nodes['root_node'] = root_node
        root_node.lower_solve(self.L0, self.L2, m=self.m, solver='l1cd',\
                              rel_tol=1e-4, mio_gap=0, int_tol = self.int_tol)
        root_node.upper_solve(self.L0, self.L2, self.m)

        # Update bounds and opt gap
        self.lower_bound = root_node.primal_value
        if (root_node.upper_bound < self.best_int):
            self.best_int = root_node.upper_bound
            self.best_beta = root_node.upper_beta
        self.initial_optimality_gap = \
            (self.best_int - self.lower_bound) / self.lower_bound
        self.optimality_gap = self.initial_optimality_gap
        self.node_counter += 1
        self.tree_stats = self.get_tree_stats()

        # Return if done
        if self.int_sol(root_node) or (self.optimality_gap <= self.gap_tol):
            return(True)
        return(False)

    def get_info(self):
        # Summary info of current state
        info = {'step_count': self.step_counter,
                'node_count': self.node_counter,
                'best_primal_value': self.best_int,
                'beta': self.best_beta,
                'lower_bound': self.lower_bound,
                'init_opt_gap' : self.initial_optimality_gap,
                'opt_gap': self.optimality_gap}
        return(info)

    def get_static_stats(self):
        # Returns static stats plus a matrix of static stats for each variable
        all_x_dot_y = np.matmul(self.x.T, self.y)
        cov = self.x.shape[0] * np.cov(self.x, rowvar=False, bias=True)

        # get quantiles from cov
        q = cov.shape[0]
        cov_flat = np.partition(cov.flatten(), kth=-q)[:-q]
        prob_stats = np.append(np.quantile(cov_flat, [0, 0.25, 0.5, 0.75, 1]), \
            np.quantile(all_x_dot_y,[0,0.25,0.5,0.75,1]))

        # for each node store dot product and cov
        p = self.x.shape[1]
        var_stats = np.zeros((p,6))
        for i in range(p):
            x_i_cov = np.partition(cov[i,:], -1)[:-1]
            var_stats[i,0:5] = np.quantile(x_i_cov,[0,0.25,0.5,0.75,1])
            var_stats[i,5] = np.dot(self.x[:,i], self.y)

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

    def get_node_stats(self, node_key):
        # Returns summary stats for a given node
        node = self.active_nodes[node_key]
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
        # Returns summary stats for branching on j in a node
        node = self.active_nodes[node_key]
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
        curr_node.lower_solve(\
            self.L0, self.L2, m=self.m, solver='l1cd', rel_tol= 1e-4, \
            mio_gap=1e-4, int_tol=self.int_tol)
        
        # Update upper bound by rounding soln
        curr_upper_bound = curr_node.upper_solve(self.L0, self.L2, self.m)
        if curr_upper_bound < self.best_int:
            self.best_int = curr_upper_bound
            self.best_beta = curr_node.upper_beta
            upper_bound_updated = True

        # Check if an integer solution
        if self.int_sol(curr_node):
            if curr_node.primal_value < self.best_int:
                self.best_int = curr_node.primal_value
                self.best_beta = curr_node.primal_beta
            del self.active_nodes[node_key]
        

    def step(self, branch_node_key, j):
        # Branches for beta_j in node branch_node_key and returns True if done
        self.step_counter += 1

        # Create two child nodes
        branch_node = self.active_nodes[branch_node_key]
        new_zlb = branch_node.zlb+[j]
        new_zub = branch_node.zub+[j]
        
        # Branch to beta_j to be 0
        node_name_1 = f'node_{self.node_counter}'
        self.active_nodes[node_name_1] = Node(parent=branch_node, zlb=new_zlb, zub=branch_node.zub, \
                                            x=branch_node.x, y=branch_node.y, xi_norm=branch_node.xi_norm)
        self.node_counter += 1

        # Branch to beta_j to be 1
        node_name_2 = f'node_{self.node_counter}'
        self.active_nodes[node_name_2] = Node(parent=branch_node, zlb=branch_node.zlb, zub=new_zub, \
                                            x=branch_node.x, y=branch_node.y, xi_norm=branch_node.xi_norm)
        del self.active_nodes[branch_node_key]
        self.node_counter += 1

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

def branch_and_bound(x, y, l0, l2, branch="max"):
    T = tree(x,y,l0,l2)
    T.start_root(None)

    fin_solving = False
    iters = 0
    num_pos = 0
    while (fin_solving == False) and (iters < MAX_ITERS):
        # Find node based on branching strategy
        if branch == "max":
            node, j = T.max_frac_branch()
        elif branch == "sample":
            frac_dict = T.get_frac_branchs()
            vals = list(frac_dict.values())
            node, j = choices(list(frac_dict.keys()), weights = vals)[0]
        elif branch == "strong_branch":
            frac_dict = T.get_frac_branchs()
            vals = frac_dict.values()
            pot_branches = choices(list(frac_dict.keys()), k=5, weights = vals)
            best_val = 0
            node, j = pot_branches[0]
            for node_key_i, j_i in pot_branches:
                T_prime = deepcopy(T)
                fin_solving, old_gap, new_gap = T_prime.step(node_key_i, j_i)
                if old_gap-new_gap > best_val:
                    best_val = old_gap-new_gap
                    node, j = node_key_i, j_i
        else:
            node = choice(list(T.active_nodes))
            j = choice(T.active_nodes[node].support)
            
        
        # Take a step
        fin_solving, old_gap, new_gap = T.step(node, j)

        # Update iterations and number positive
        if old_gap > new_gap:
            num_pos += 1
        iters += 1
    return(iters, num_pos)
        
#x = np.loadtxt("/Users/alice/Dropbox/var_selection/synthetic_data/batch_2/x_gen_syn_n3_p50_corr0.5_snr5.0_seed2022_394.csv", delimiter = ",")
#y = np.loadtxt("/Users/alice/Dropbox/var_selection/synthetic_data/batch_2/y_gen_syn_n3_p50_corr0.5_snr5.0_seed2022_394.csv", delimiter=",")
#l0 = 0.6
#l2 = 0.0     
#iters, num_pos = branch_and_bound(x, y, l0, l2, "strong_branch")
#print(iters, num_pos)
