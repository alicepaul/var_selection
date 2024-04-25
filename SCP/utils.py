from tree import tree
from model import Agent
from SCproblem import Problem

def filepathstrlist(filepath, start = 0, end = 80):
    file_paths = []
    for i in range(start,end):
        file_paths.append(filepath + str(i) + ".lp")
    return file_paths


def parse_test_instance_from_file(file_path):
    with open(file_path, 'r') as file:
        input_text = file.read()

    lines = input_text.split("\n")
    num_variables = 0
    num_constraints = 0
    constraints = []
    variables = []

    for line in lines:
        if "Variables" in line:
            num_variables = int(line.split()[3])
            variables = [0] * num_variables  
        elif "Constraints" in line:
            num_constraints = int(line.split()[3])
        elif line.startswith(" c"):
            parts = line.split(":")[1].split(">=")[0].strip().split(" +")
            constraint = [0] * num_variables
            for part in parts:
                if part:
                    index = int(part.split("x")[1])
                    constraint[index] = 1
            constraints.append(constraint)
    
    # Check if data was parsed correctly
    if len(constraints) != num_constraints or len(variables) != num_variables:
        print("Error parsing the test instance.")
        return None, None
    
    return constraints

def load_problem_instances(file_paths):
    problem_instances = []
    for file_path in file_paths:
        x = parse_test_instance_from_file(file_path)
        problem_instances.append(Problem(x))
    return problem_instances

def solve_bnb(problem_instances, filename = "bnb_results_50_100.txt"):
    bb_metrics = []
    iterationsOnly = []
    i = 0
    for instance in problem_instances:
        newTree = tree(instance)
        iterations, _, opt_gap = newTree.branch_and_bound()
        # Store the branch and bound metrics for the instance
        bb_metrics.append((iterations, opt_gap))
        iterationsOnly.append(iterations)
        print("Solved instance ", i," in ", iterations, " iterations.")
        with open(filename, "a") as file:
            file.write(f"Solved instance: {i} in {iterations} iterations.\n")
        file.close()
        i += 1
    
    return bb_metrics, iterationsOnly

def train_model(problem_instances, modelParams= 22, filename = "training_metrics.txt"):
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
