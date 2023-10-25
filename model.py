# Toby Dekara and Alice Paul
# Created July 19, 2022
# Model for RL agent

# Adapted from Human-Level Control Through Deep Reinforcement Learning
#Copyright (c) 2018 Algonomicon LLC

import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import tree 
from tree import get_state_pairs
from settings import MAX_ITERS, EPSILON_START, \
     EPSILON_END, EPSILON_DECAY, BATCH_SIZE, INT_EPS, GAMMA, TARGET_UPDATE

# Memory representation of states
Transition = namedtuple('Transition', 
                        ('prev_state', 'state', 'reward'))

# Deep Q Network
class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        output1 = F.relu(self.fc1(x))
        output2 = F.relu(self.fc2(output1))
        output = F.relu(self.fc3(output2))
        return(output)

# Memory for our agent
class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Agent that performs, remembers and learns actions
class Agent():
    def __init__(self):
        self.policy_net = DQN(32) # TODO: update sizes
        self.target_net = DQN(32)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = Memory(10000)
        self.episodes_played = 0
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_end = EPSILON_END

    def retrobranch(self, tree):
        # Complete Tree -- Get (Non-Optimal) States for Leaf Nodes
        for node in tree.all_nodes.values():
            if node.state is None:
                support = node.support
                best_val = -math.inf
                best_j = 0
                for i in range(len(support)):
                    state = torch.tensor(np.array([tree.get_state(node.node_key, support[i])]), 
                                         dtype=torch.float)
                    # Agent estimates using policy network
                    val = self.policy_net(state) 
                    if val > best_val:
                        best_val = val
                        best_j = support[i]
            
                node.state = tree.get_state(node.node_key, best_j)

        # Set rewards
        total_reward = 0

        # Call tree function to create all state to state pairs
        state_pairs = get_state_pairs(tree.root)
        for prev, curr, r in state_pairs:
            total_reward += r

            # Add state pairs and reward to memory 
            self.memory.push(torch.tensor(np.array([prev]), dtype=torch.float), 
                             torch.tensor(np.array([curr]), dtype=torch.float), 
                             torch.tensor([r], dtype=torch.float))
        
        # Update target network
        if self.episodes_played % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return total_reward

    def select_action(self, T):
        # Select an action according to an epsilon greedy approach        
        if (random.random() < self.epsilon):
            # max fraction branching
            best_node_key, best_j = T.max_frac_branch()
        else:
            # calculate estimated value for all nodes
            best_val = -math.inf
            best_node_key = None
            best_j = 0

            for node_key in T.active_nodes:
                support = T.active_nodes[node_key].support
                for i in range(len(support)):
                    if (T.active_nodes[node_key].z[i] < INT_EPS) or (T.active_nodes[node_key].z[i] > 1-INT_EPS):
                        continue
                    state = torch.tensor(np.array([T.get_state(node_key, support[i])]), dtype=torch.float)
                    # Agent estimates usings policy network
                    val = self.policy_net(state) 
                    if(val > best_val):
                        best_val = val
                        best_node_key = node_key
                        best_j = support[i]

        return(best_node_key, best_j)
    
    def replay_memory(self):
        # Only Replay Memory if enough enteries in Memory
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample from our memory
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Concatenate our tensors for previous states
        prev_state_batch = torch.cat(batch.prev_state)
        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)

        # Predict Q-values for the previous states
        pred_q_values = self.policy_net(prev_state_batch)
        pred_q_values = pred_q_values.squeeze(1) # Match shape of targets

        # Compute expected Q-values based on next states and rewards
        with torch.no_grad():
            max_next_q_values = torch.flatten(self.target_net(state_batch))
            targets = reward_batch + GAMMA * max_next_q_values

        # Compute loss
        loss_f = nn.MSELoss()
        loss = loss_f(pred_q_values, targets)

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        # Update Parameters
        self.optimizer.step()


def RL_solve(agent, x, y, l0, l2):
    # Solving an instance using agent to make choices in tree
    T = tree.tree(x,y,l0,l2)
    fin_solving = T.start_root(None)
    iters = 0

    while (fin_solving == False) and (iters < MAX_ITERS):
        # Select and perform an action
        node, j = agent.select_action(T)
        fin_solving, old_gap, new_gap = T.step(node, j) 

        # Optimize the target network using replay memory
        agent.replay_memory()

        iters += 1

    # Store tree in memory and get total reward for tree
    tot_reward = agent.retrobranch(T)

    # Update number of episodes Agent has played
    agent.episodes_played += 1
        
    return(iters, tot_reward, T)