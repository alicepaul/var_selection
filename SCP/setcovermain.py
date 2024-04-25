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

filepath = "SCP/ecole_tests/sc-100-100-0.08/sc-100-100-"
file_paths = utils.filepathstrlist(filepath, 80, 100)
problem_instances = utils.load_problem_instances(file_paths)
# # bb_metrics, iterationsOnly = utils.solve_bnb(problem_instances)
# trained_agent, trainingmetrics, RLiterationsOnly = utils.train_model(problem_instances)
# print(RLiterationsOnly)
# trained_agent.save_model("SCP/saved_models/sc-50-100.pth")
agent = Agent(22)  # Ensure you have the correct input size for your network
agent.load_model("SCP/saved_models/sc-100-100-short.pth")
rl_metrics, iterationsOnly = utils.evaluate_model(agent, problem_instances)
