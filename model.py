# Toby Dekara and Alice Paul
# Created July 19, 2022
# Model for RL agent

# Adapted from Human-Level Control Through Deep Reinforcement Learning
#Copyright (c) 2018 Algonomicon LLC

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import tree
from settings import REWARD_TYPE, MAX_ITERS, EPSILON_START, \
     EPSILON_END, EPSILON_DECAY, BATCH_SIZE, INT_EPS, GAMMA

# Memory representation of states
Transition = namedtuple('Transition', 
                        ('prev_state', 'state', 'reward'))

# Deep Q Network
class DQN(nn.Module):
    def __init__(self, input_size, reward_type = "cont"):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.reward = reward_type

    def forward(self, x):
        output1 = F.relu(self.fc1(x))
        output2 = F.relu(self.fc2(output1))
        output = F.relu(self.fc3(output2))
        if (self.reward == "bin"):
            output = torch.sigmoid(output)
        return(output)

# Memory representation for our agent
class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Agent that performs, remembers and learns actions
class Agent():
    def __init__(self, reward_type = "cont"):
        self.policy_net = DQN(32, reward_type) # TODO: update sizes
        self.target_net = DQN(32, reward_type)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = Memory(10000)
        self.steps_done = 0
        self.reward = reward_type

    def remember(self, *args):
        self.memory.push(*args)

    def select_action(self, T):
        # Select an action according to an epsilon greedy approach
        sample = random.random()
        epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if (sample < epsilon_threshold):
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
                    state = torch.tensor([T.get_state(node_key, support[i])], dtype=torch.float)
                    # Agent estimates usings policy network
                    val = self.policy_net(state) 
                    if(val > best_val):
                        best_val = val
                        best_node_key = node_key
                        best_j = support[i]

        return(T.get_state(best_node_key, best_j), best_node_key, best_j)
    
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
        
        # Update Parameters
        self.optimizer.step()


def RL_solve(agent, x, y, l0, l2):
    # Solving an instance using agent to make choices in tree
    T = tree.tree(x,y,l0,l2)
    fin_solving = T.start_root(None)
    iters = 0
    tot_rewards = 0

    # Set Beginning State
    prev_state = torch.tensor([T.get_state('root_node', 0)], dtype=torch.float)

    while (fin_solving == False) and (iters < MAX_ITERS):
        # Select and perform an action
        state, node, j = agent.select_action(T)
        state = torch.tensor([state], dtype=torch.float)
        fin_solving, old_gap, new_gap = T.step(node, j) # Add done to Transitions

        # Calculate reward
        bin_reward = 1
        if old_gap == new_gap:
            bin_reward = 0

        reward = bin_reward
        if (REWARD_TYPE == "cont"):
            reward = (old_gap-new_gap)/T.initial_optimality_gap

        tot_rewards += bin_reward 
        reward = torch.tensor([reward], dtype=torch.float)
    
        # Store the transition in memory
        agent.remember(prev_state, state, reward)

        # Set Previous State
        prev_state = state

        # Optimize the target network using replay memory
        agent.replay_memory()

        iters += 1
        
    return(iters, tot_rewards, len(T.best_beta))



