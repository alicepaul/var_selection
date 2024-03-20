import Model
import os
import numpy as np
import pandas as pd
from random import shuffle
import Tree
import pickle
import torch
from itertools import count
from settings import DATA_BATCH, MOD_NUM
from sklearn.linear_model import LinearRegression
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Directory of problems
my_path = "synthetic_data/batch_"+str(DATA_BATCH)
files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f)) and f[0]=="x"]
shuffle(files)
num_files = 0

# Results
column_names = ["data", "num_file", "L0", "L2", "Epsilon", 
                "RL_iters", "RL_rewards", "RL_nnz", "RL_OG",
                "MF_iters", "MF_nnz", "MF_OG",
                "SB_iters", "SB_nnz", "SB_OG"]
res = pd.DataFrame(columns = column_names)

# Agent RL
agent = Model.Agent(34)

# Load Model
# agent.policy_net.load_state_dict(torch.load(f"synthetic_data/models/mixed_model.pt"))
# agent.epsilon = 0.05

for f in files:
    x_file = os.path.join(my_path, f)
    y_file = os.path.join(my_path, "y"+f[1:len(f)])
    x = np.loadtxt(x_file, delimiter = ",")
    y = np.loadtxt(y_file, delimiter=",")
    l0_max = max(abs(np.dot(x.T, y)))
    l0_range = [.3, .15, .1, .05]
    l2 = 0.0

    # Find Optimal M value
    linear_model = LinearRegression()
    linear_model.fit(x, y)
    max_abs_beta = np.max(np.abs(linear_model.coef_)) # Calculate the Maximum Absolute Coef
    m = 1.5 * max_abs_beta

    print(f'File: {num_files}, p = {x.shape[1]}, m = {m}, l0_max = {l0_max}', flush=True)

    for val in l0_range:
        l0 = l0_max * val
    
        # Max Frac
        p = Tree.Problem(x,y,l0,l2, m)
        tree = Tree.tree(p)
        MF_iters, MF_nnz, MF_og = tree.branch_and_bound("max")

        # Strong Branch 
        p = Tree.Problem(x,y,l0,l2, m)
        tree = Tree.tree(p)
        SB_iters, SB_nnz, SB_og = 0,0,0 # tree.branch_and_bound("strong_branch")

        # RL
        p = Tree.Problem(x,y,l0,l2, m)
        tree = Tree.tree(p)
        RL_iters, RL_rewards, RL_nnz, RL_og = agent.RL_solve(tree,training=False)

        # Add results to file
        data = [[f, num_files, l0, l2, agent.epsilon,
                RL_iters, RL_rewards, RL_nnz, RL_og,
                MF_iters, MF_nnz, MF_og,
                SB_iters, SB_nnz, SB_og]]
        new_row = pd.DataFrame(data=data, columns=column_names)
        res = pd.concat([res, new_row], ignore_index=True)
    num_files += 1

# Save Results
res.to_csv(f"synthetic_data/testing"+str(DATA_BATCH)+".csv", index=False)

# Save Model Information
# torch.save(agent.policy_net.state_dict(), f"synthetic_data/models/model_pn_{MOD_NUM}.pt")    # Save Policy Net