import utils
import torch
import argparse
from model import Agent
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model and save it.')
parser.add_argument('--model-location', type=str, help='Filename to save the trained model', default="saved_models/sc-100-100-0.10.pt")
parser.add_argument('--instance-location', type=str, help='Location of the instances', default="saved_models/sc-100-100-0.10.pt")
parser.add_argument('--instance-start', type=int, help='instance start #', default=0)
parser.add_argument('--instance-end', type=int, help='instance end #', default=61)
args = parser.parse_args()

filepath = "maxcut-tests/25vars/graph_"
file_paths = utils.filepathstrlist(filepath)
problem_instances = utils.load_problem_instances(file_paths)
# bb_metrics, iterationsOnly = utils.solve_bnb(problem_instances)
trained_agent, trainingmetrics, RLiterationsOnly = utils.train_model(problem_instances)
print(RLiterationsOnly)
trained_agent.save_model("saved_models/maxcut-25.pth")
# agent = Agent(22)  # Ensure you have the correct input size for your network
# agent.load_model("SCP/saved_models/sc-100-100-short.pth")
# rl_metrics, iterationsOnly = utils.evaluate_model(agent, problem_instances)
