import Tree
import os
import numpy as np
import pandas as pd
from random import shuffle
import pickle
from settings import DATA_BATCH
import random

np.random.seed(0)
random.seed(0)

# Directory of problems
my_path = "synthetic_data/batch_"+str(DATA_BATCH)
files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f)) and f[0]=="x"]
shuffle(files)
num_files = 0

# Containers for tree data
max_frac_trees = []
strong_branch_trees = []

for f in files:
    print(num_files, flush=True)
    x_file = os.path.join(my_path, f)
    y_file = os.path.join(my_path, "y"+f[1:])
    x = np.loadtxt(x_file, delimiter = ",")
    y = np.loadtxt(y_file, delimiter=",")
    l0_max = max(abs(np.dot(x.T, y)))/2.0
    l0 = l0_max*0.3 
    l2 = 0.0  
    m = 0.61 

    # Solve with max fraction branch
    p_max = Tree.Problem(x, y, l0, l2, m)
    tree_max = Tree.tree(p_max)
    tree_max.branch_and_bound("max")
    max_frac_trees.append(tree_max)

    # Solve with strong branch
    p_strong = Tree.Problem(x, y, l0, l2, m)
    tree_strong = Tree.tree(p_strong)
    tree_strong.branch_and_bound("strong_branch")
    strong_branch_trees.append(tree_strong)

    num_files += 1

# Save tree data to files
with open('synthetic_data/trees/max_frac_trees.pkl', 'wb') as f:
    pickle.dump(max_frac_trees, f)

with open('synthetic_data/trees/strong_branch_trees.pkl', 'wb') as f:
    pickle.dump(strong_branch_trees, f)