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
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Load max_frac_trees
with open('synthetic_data/trees/max_frac_trees.pkl', 'rb') as f:
    max_frac_trees = pickle.load(f)

# Load strong_branch_trees
with open('synthetic_data/trees/strong_branch_trees.pkl', 'rb') as f:
    strong_branch_trees = pickle.load(f)

# Initialize Agents
mx_agent = Model.Agent()
sb_agent = Model.Agent()

# Train Agent on Max Fraction Branch Completed Episodes
print('Start Training', flush=True)
for tree in max_frac_trees:
    mx_agent.retrobranch(tree)

    for i in range(128):
        mx_agent.replay_memory()
        
print('Completed Training RL-MX', flush=True)

# Train Agent on Strong Branch Completed Episodes
for tree in strong_branch_trees:
    sb_agent.retrobranch(tree)

    for i in range(128):
        sb_agent.replay_memory()

print('Completed Training RL-SB', flush=True)


### Evaluate Trained Models ###

# Directory of problems
my_path = "synthetic_data/batch_"+str(DATA_BATCH)
files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f)) and f[0]=="x"]
shuffle(files)
num_files = 0

# Results
column_names = ["data", "num_file", "L0", "L2", 
                "RL-MX_iters", "RL-MX_rewards", "RL-MX_nnz", "RL-MX_OG",
                "RL-SB_iters", "RL-SB_rewards", "RL-SB_nnz", "RL-SB_OG",
                "MF_iters", "MF_nnz", "MF_OG"]
res = pd.DataFrame(columns = column_names)


for f in files:
    print(num_files, flush=True)
    x_file = os.path.join(my_path, f)
    y_file = os.path.join(my_path, "y"+f[1:len(f)])
    x = np.loadtxt(x_file, delimiter = ",")
    y = np.loadtxt(y_file, delimiter=",")
    l0_max = max(abs(np.dot(x.T, y)))/2.0
    l0 = 0.01 # l0_max*0.3
    l2 = 0.0
    # Optimal m value is calculated prior in tuning.ipynb
    m = 1.34 # 0.61

    # Solve with agent and branch and bound directly
    # Max Frac
    p = Tree.Problem(x,y,l0,l2, m)
    tree = Tree.tree(p)
    MF_iters, MF_nnz, MF_og = tree.branch_and_bound("max")

    # RL
    RL_MX_iters, RL_MX_rewards, RL_MX_nnz, RL_MX_og = mx_agent.RL_solve(x,y,l0,l2, m, training=False)
    RL_SB_iters, RL_SB_rewards, RL_SB_nnz, RL_SB_og = sb_agent.RL_solve(x,y,l0,l2, m, training=False)

    # Add results to file
    data = [[f, num_files, l0, l2,
            RL_MX_iters, RL_MX_rewards, RL_MX_nnz, RL_MX_og,
            RL_SB_iters, RL_SB_rewards, RL_SB_nnz, RL_SB_og,
            MF_iters, MF_nnz, MF_og]]
    new_row = pd.DataFrame(data=data, columns=column_names)
    res = pd.concat([res, new_row], ignore_index=True)
    num_files += 1

# Save Results
res.to_csv("synthetic_data/comparison_"+str(DATA_BATCH)+".csv", index=False)

# Save Model Information
torch.save(mx_agent.policy_net.state_dict(), f"synthetic_data/models/model_mx_{MOD_NUM+1}.pt")    # Save Policy Net
torch.save(sb_agent.policy_net.state_dict(), f"synthetic_data/models/model_sb_{MOD_NUM+1}.pt")    # Save Policy Net