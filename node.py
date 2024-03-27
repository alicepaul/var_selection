# Taken from https://github.com/alisaab/l0bnb/tree/master

from scipy import optimize as sci_opt
import numpy as np


class Node:
    def __init__(self, parent, node_key, zlb: list, zub: list):
        """
        Initialize a Node

        Parameters
        ----------
        parent: Node or None
            the parent Node
        
        node_key: Str
            Name associated with Node, used for Node Lookup

        """
        
        self.parent_dual = parent.dual_value if parent else None
        self.parent_primal = parent.primal_value if parent else None

        if parent and parent.warm_start:
            self.warm_start = \
                {i: j for i, j in zip(parent.support, parent.primal_beta)}
        else:
            self.warm_start = None

        self.level = parent.level + 1 if parent else 0
        self.zlb = zlb
        self.zub = zub
        self.z = None

        self.upper_bound = None
        self.primal_value = None
        self.dual_value = None

        self.support = None
        self.upper_z = None
        self.primal_beta = None

        # Each node will store it's children's nodes and current state
        self.node_key = node_key
        self.opt_gap = 0
        self.parent_key = None
        self.is_leaf = True
        self.left = None
        self.right = None
        self.state = None

    def get_info(self):
        return f'node_key: {self.node_key}, is_leaf: {self.is_leaf}, ' \
            f'state: {self.state}'
    
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

    def __str__(self):
        return f'level: {self.level}, lower bound: {self.primal_value}, ' \
            f'upper bound: {self.upper_bound}'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.level == other.level:
            if self.primal_value is None and other.primal_value:
                return True
            if other.primal_value is None and self.primal_value:
                return False
            elif not self.primal_value and not other.primal_value:
                return self.parent_primal > \
                       other.parent_cost
            return self.primal_value > other.lower_bound_value
        return self.level < other.level