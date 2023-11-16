import model
import os
import numpy as np
import pandas as pd
from random import shuffle
import tree
import torch
from itertools import count
from settings import TARGET_UPDATE, DATA_BATCH, MAX_ITERS

# Directory of problems
my_path = "synthetic_data/batch_"+str(DATA_BATCH)
files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f)) and f[0]=="x"]
shuffle(files)
num_files = 0

# Results
column_names = ["data", "num_file", "L0", "L2", "NNZ", 
                "RL_iters", "RL_rewards", 
                "MF_iters", "MF_rewards",
                "SB_iters", "SB_rewards"]
res = pd.DataFrame(columns = column_names)

# Agent RL
agent = model.Agent()

for f in files:
    print(f)
    print(num_files)
    x_file = os.path.join(my_path, f)
    y_file = os.path.join(my_path, "y"+f[1:len(f)])
    x = np.loadtxt(x_file, delimiter = ",")
    y = np.loadtxt(y_file, delimiter=",")
    l0_max = max(abs(np.dot(x.T, y)))/2.0
    l0_grid = [0.0001, 0.001, 0.01, 0.1]
    l2 = 0.0

    # Iterate through values of l0
    for alpha in l0_grid:
        l0 = alpha*l0_max

        # Solve with agent and branch and bound directly
        RL_iters, RL_rewards, nnz = model.RL_solve(agent, x, y, l0, l2)
        MF_iters, MF_rewards = tree.branch_and_bound(x, y, l0, l2, "max")
        SB_iters, SB_rewards = tree.branch_and_bound(x, y, l0, l2, "strong_branch")

        # Add results to file
        data = [[f, num_files, l0, l2, nnz,
                 RL_iters, RL_rewards,
                 MF_iters, MF_rewards,
                 SB_iters, SB_rewards]]
        new_row = pd.DataFrame(data=data, columns=column_names)
        res = pd.concat([res, new_row], ignore_index=True)
    num_files += 1

res.to_csv("synthetic_data/results_"+str(DATA_BATCH)+".csv", index=False)
# torch.save(agent.policy_net.state_dict(), "synthetic_data/model_"+str(DATA_BATCH)+".pt")
