from tree import tree
from model import Agent
from MCproblem import Problem
import numpy as np

def filepathstrlist(filepath, start = 1, end = 60):
    file_paths = []
    for i in range(start,end):
        file_paths.append(filepath + str(i) + ".txt")
    return file_paths

def parse_test_instance_from_file(file_path):
    return np.loadtxt(file_path, dtype=int)

def load_problem_instances(file_paths):
    problem_instances = []
    for file_path in file_paths:
        x = parse_test_instance_from_file(file_path)
        problem_instances.append(Problem(x))
    return problem_instances

def solve_bnb(problem_instances, filename = "bnb_results_30.txt"):
    bb_metrics = []
    iterationsOnly = []
    i = 0
    for instance in problem_instances:
        newTree = tree(instance)
        iterations, _, opt_gap = newTree.branch_and_bound()
        bb_metrics.append((iterations, opt_gap))
        iterationsOnly.append(iterations)
        print("Solved instance ", i," in ", iterations, " iterations.")
        with open(filename, "a") as file:
            file.write(f"Solved instance: {i} in {iterations} iterations.\n")
        file.close()
        i += 1
    
    return bb_metrics, iterationsOnly

def train_model(problem_instances, modelParams= 31, filename = "training_metrics_25.txt"):
    agent = Agent(modelParams)  # Initialize the RL agent with the appropriate input size
    training_metrics = []
    iterationsOnly = []
    epsilons = []
    i = 0

    for instance in problem_instances:
        newTree = tree(instance)
        iterations, total_reward, candidate_sol_length, opt_gap = agent.RL_solve(newTree, training=True)
        training_metrics.append((iterations, total_reward, candidate_sol_length, opt_gap))
        iterationsOnly.append(iterations)
        epsilons.append(agent.epsilon)
        print(f"Episode: {i} Iterations={iterations}, Total Reward={total_reward}, Epsilon={agent.epsilon}, Optimality Gap={opt_gap}")
        with open(filename, "a") as file:
            file.write(f"Episode: {i} Iterations={iterations}, Total Reward={total_reward}, Epsilon={agent.epsilon}, Optimality Gap={opt_gap}\n")
        file.close()
        i += 1
    return agent, training_metrics, iterationsOnly
    
    
def evaluate_model(agent, test_instances, filename = "eval_50_100-large.txt"):
    rl_metrics = []
    iterationsOnly = []
    epsilons = []
    i = 0
    for instance in test_instances:
        newTree = tree(instance)
        iterations, total_reward, candidate_sol_length, opt_gap = agent.RL_solve(newTree, training=False)
        epsilons.append(agent.epsilon)
        # Store the evaluation metrics for the test instance
        rl_metrics.append((iterations, total_reward, opt_gap))
        iterationsOnly.append(iterations)
        print(f"Episode: {i} Iterations={iterations}, Total Reward={total_reward},Epsilon={agent.epsilon}, Optimality Gap={opt_gap}")
        with open(filename, "a") as file:
            file.write(f"Episode: {i} Iterations={iterations}, Total Reward={total_reward}, Epsilon={agent.epsilon}, Optimality Gap={opt_gap}\n")
        file.close()
        i += 1
    return rl_metrics, iterationsOnly
