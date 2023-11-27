# Reinforcement Learning for Best Subset Variable Selection

This repository contains ongoing work focusing on "Reinforcement Learning for Best Subset Variable Selection". Our current work builds on the research of Tobias DeKara, which used a novel approach to best subset selection for sparse linear regression using mixed integer quadratic optimization (MIQO) in combination with reinforcement learning (RL) to enhance the efficiency of solving these MIQO problems/ Our current phase of research, undertaken by Kameel Dossal and Shreyas Mishra and advised by Professor **Alice Paul**, seeks to further these advancements. Our focus remains on the innovative application of reinforcement learning to optimize the solving of MIQO problems, particularly in the context of variable selection. 


## Repository Structure
- `synthetic_data/`: Contains generated data for variable selection, stored models, and results.
- `SCP/`: Related files for set-covering problem, using shared problem structure and class for variable selection.
- `gen_syn_data.py`: Script for generating synthetic data for variable selection.
- `main.py`: Main script to run algorithms, including the RL agent, on synthetic data. Outputs stored in synthetic_data/results and models in synthetic_data/models.
- `Node.py`: Code for the Node class/data structure within the tree.
- `Tree.py`: Contains the problem and tree classes used for the variable selection task.
- `Tuning.ipynb`: Jupyter Notebook for model tuning, exploring the B&B algorithm, and evaluating results.
- `settings.py`: Configuration file for specifying hyperparameters, dataset selection, and storage options during training.

## How to Use
1. Clone the repository.
2. Install required dependencies.
3. Run gen_syn_data.py to generate synthetic data.
4. Tune settings in setting.py.
5. Use main.py to execute the algorithms and generate results.