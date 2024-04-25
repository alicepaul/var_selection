import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from settings import MAX_ITERS, EPSILON_START, \
     EPSILON_END, EPSILON_DECAY, BATCH_SIZE, INT_EPS, GAMMA, TARGET_UPDATE


# Memory representation of states
Transition = namedtuple('Transition', 
                        ('prev_state', 'state', 'reward'))

# Deep Q Network
class DQN(nn.Module):
    def __init__(self, input_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        output1 = F.leaky_relu(self.fc1(x))
        output2 = F.leaky_relu(self.fc2(output1))
        output = self.fc3(output2)
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
    def __init__(self, n):
        self.policy_net = DQN(n)
        self.target_net = DQN(n)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = Memory(10000)
        self.episodes_played = 0
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_end = EPSILON_END

    def retrobranch(self, tree):
        """
        Complete all leaf states in finished tree, then store all state pairs in memory
        Uses max_frac_branch for completing tree for efficiency
        Checks if episodes_played divisible by TARGET_UPDATE, if so updates target network

        Parameters
        ----------
        tree: Tree Class
            the finished tree for retrobranching
        
        total_reward: Int
            total reward for playing through tree
        """

        # Complete Tree -- Get (Non-Optimal) States for Leaf Nodes
        for node in tree.all_nodes.values():
            if node.state is None: 
                if len(node.support) == 0:
                    best_j = 0
                # Select a best_j 
                else:
                    # z = node.z
                    # support = node.support
                    # diff = [min(1-z[i], z[i]-0) for i in range(len(support))]
                    # best_j = support[np.argmax(diff)]
                    if node.z is None:
                        if len(node.support) != 0:
                            best_j = node.support[0]
                    else:
                        z = [node.z[i] for i in node.support]
                        diff = [min(1 - val, val) for val in z]
                        max_diff = max(diff)
                        potential_j_index = diff.index(max_diff)
                        best_j = node.support[potential_j_index]
            
                node.state = tree.get_state(node.node_key, best_j)

        # Set rewards
        total_reward = 0

        # Call tree function to create all state to state pairs
        state_pairs = tree.get_state_pairs(tree.root)

        # Sample 128 state pairs if there are more than 128, otherwise use all
        sample_size = 128
        if len(state_pairs) > sample_size:
            sampled_pairs = random.sample(state_pairs, sample_size)
        else:
            sampled_pairs = state_pairs
        
        for prev, curr, r in sampled_pairs:
            
            total_reward += r
            curr_list = curr.tolist()
            if None in curr:
                print("curr", type(curr), curr)
            if None in prev:
                print("prev",type(prev), prev)
            
            # # Add state pairs and reward to memory 
            # self.memory.push(torch.tensor(np.array([prev]), dtype=torch.float), 
            #                  torch.tensor(np.array([curr]), dtype=torch.float), 
            #                  torch.tensor([r], dtype=torch.float))
            # Convert the array to a list and replace None with a suitable numerical value
            
            curr_list = [0.0 if x is None else x for x in curr_list]

            # Convert the modified list back to a NumPy array
            curr_array = np.array([curr_list])

            # Add state pairs and reward to memory
            self.memory.push(torch.tensor(np.array([prev]), dtype=torch.float),
                            torch.tensor(curr_array, dtype=torch.float),
                            torch.tensor([r], dtype=torch.float))
        
        # Update target network
        if self.episodes_played % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return total_reward

    def select_action(self, T):
        """
        Select an optimal node and j to split on given a tree

        Parameters
        ----------
        T: Tree Class
            the tree to select an action on
        
        best_node_key: Str
            string tied to the node to split on
        best_j: Int
            int representing the variable to split on
        """

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
        """
        Samples a BATCH_SIZE number of items from memory, optimizes and updates policy network
        """

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


    def RL_solve(self, T, training=True): # Give it a Tree
        # Solving an instance using agent to make choices in tree
        fin_solving = T.start_root()
        if fin_solving:
            print("Optimal solution found at the root node.")
            return 0, 1, len(T.candidate_sol), T.optimality_gap
        # fin_solving = T.start_root()
        iters = 0
        tot_reward = 0

        while (fin_solving == False) and (iters < 8000):
            # Select and perform an action
            node, j = self.select_action(T)
            fin_solving, old_gap, new_gap = T.step(node, j) 

            iters += 1
        tot_reward = self.retrobranch(T)
        
        if training:
            # Store tree in memory and get total reward for tree
            

            # Optimize the target network using replay memory 
            # 128 iters after each episode
            for i in range(8):
                self.replay_memory()

            # Update number of episodes Agent has played
            self.episodes_played += 1
            
        return iters, tot_reward, len(T.candidate_sol), T.optimality_gap
    
    def save_model(self, file_name):
        torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episodes_played': self.episodes_played,
                'epsilon': self.epsilon}, file_name)

    def load_model(self, file_name):
        checkpoint = torch.load(file_name)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episodes_played = checkpoint['episodes_played']
        self.epsilon = checkpoint['epsilon']
        # Ensure target net and policy net are in the same state
        self.target_net.load_state_dict(self.policy_net.state_dict())
