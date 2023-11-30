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

# Directory of problems
my_path = "synthetic_data/batch_"+str(DATA_BATCH)
files = [f for f in os.listdir(my_path) if os.path.isfile(os.path.join(my_path,f)) and f[0]=="x"]
shuffle(files)
num_files = 0

# Results
column_names = ["data", "num_file", "L0", "L2", "Epsilon", 
                "RL_iters", "RL_rewards", "RL_nnz", "RL_OG",
                "MF_iters", "MF_rewards", "MF_nnz", "MF_OG"]
res = pd.DataFrame(columns = column_names)

# Agent RL
agent = Model.Agent()

# Load Previous Model
# agent.policy_net.load_state_dict(torch.load(f"/users/kdossal/synthetic_data/models/{MOD_NUM}.pt"))
# agent.target_net.load_state_dict(torch.load(f"/users/kdossal/synthetic_data/models/{MOD_NUM}.pt"))
# agent.optimizer.load_state_dict(torch.load(f"synthetic_data/models/optimizer_{MOD_NUM}.pt"))

# with open(f'synthetic_data/models/memory_{MOD_NUM}.pkl', 'rb') as f:
#     agent.memory = pickle.load(f)

# agent.epsilon = ??? # Set using most recent results or to desired values

for f in files:
    print(num_files, flush=True)
    x_file = os.path.join(my_path, f)
    y_file = os.path.join(my_path, "y"+f[1:len(f)])
    x = np.loadtxt(x_file, delimiter = ",")
    y = np.loadtxt(y_file, delimiter=",")
    l0_max = max(abs(np.dot(x.T, y)))/2.0
    l0 = 0.01 # l0_max*0.3
    l2 = 0.0

    # Solve with agent and branch and bound directly
    # Optimal m value is calculated prior in tuning.ipynb
    RL_iters, RL_rewards, RL_nnz, RL_og = agent.RL_solve(x,y,l0,l2, m = 1.3)

    p = Tree.Problem(x,y,l0,l2, m = 1.3)
    tree = Tree.tree(p)
    MF_iters, MF_rewards, MF_nnz, MF_og = tree.branch_and_bound("max")

    # Add results to file
    data = [[f, num_files, l0, l2, agent.epsilon,
            RL_iters, RL_rewards, RL_nnz, RL_og,
            MF_iters, MF_rewards, MF_nnz, MF_og]]
    new_row = pd.DataFrame(data=data, columns=column_names)
    res = pd.concat([res, new_row], ignore_index=True)
    num_files += 1

# Save Results
res.to_csv("synthetic_data/results_"+str(DATA_BATCH)+".csv", index=False)

# # Save Model Information
# torch.save(agent.policy_net.state_dict(), f"synthetic_data/models/model_pn_{MOD_NUM+1}.pt")    # Save Policy Net
# torch.save(agent.target_net.state_dict(), f"synthetic_data/models/model_tn_{MOD_NUM+1}.pt")    # Save Target Net
# torch.save(agent.optimizer.state_dict(), f"synthetic_data/models/optimizer_{MOD_NUM+1}.pt")    # Save Optimizer

# # Save memory
# with open(f'synthetic_data/models/memory_{MOD_NUM+1}.pkl', 'wb') as f:
#     pickle.dump(agent.memory, f) # Save Memory