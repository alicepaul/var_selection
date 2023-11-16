import numpy as np
import random
import os
from datetime import datetime

def generate_scp_dataset(num_universe_items, num_sets, min_set_size, max_set_size):
    """
    Generate a dataset for the Set Covering Problem.

    :param num_universe_items: Number of items in the universe.
    :param num_sets: Number of sets to generate.
    :param min_set_size: Minimum number of items in each set.
    :param max_set_size: Maximum number of items in each set.
    :return: A tuple (universe, x) where universe is a set of all items,
             and x is a binary matrix representing the sets.
    """
    # Generate the universe of items
    universe = set(range(num_universe_items))

    # Initialize the binary matrix for sets
    x = np.zeros((num_sets, num_universe_items), dtype=int)

    # Generate the sets and update the binary matrix
    for i in range(num_sets):
        set_size = random.randint(min_set_size, max_set_size)
        set_items = random.sample(list(universe), set_size)  # Convert the set to a list here
        for item in set_items:
            x[i, item] = 1

    # Ensure every item is covered at least once
    uncovered_items = universe.copy()
    for item in universe:
        if not np.any(x[:, item]):
            random_set = random.randint(0, num_sets - 1)
            x[random_set, item] = 1
            uncovered_items.discard(item)

    # If there are still uncovered items, add them to random sets
    while uncovered_items:
        item = uncovered_items.pop()
        random_set = random.randint(0, num_sets - 1)
        x[random_set, item] = 1

    return universe, x

# Usage
num_universe_items = 100  # Total number of items in the universe
num_sets = 50            # Total number of sets
min_set_size = 5         # Minimum number of items in each set
max_set_size = 30        # Maximum number of items in each set

universe, x = generate_scp_dataset(num_universe_items, num_sets, min_set_size, max_set_size)

# Output the generated instance
print("Universe:", universe)
print("Binary Matrix X:")
print(x)

# Create the 'data' directory if it doesn't exist
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Get the current date and time for the filename
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"scp_dataset_{current_time}.csv"
x_file_path = os.path.join(data_dir, filename)

# Save the binary matrix 'x' to a file in the 'data' directory
np.savetxt(x_file_path, x, delimiter=",", fmt='%d')

print(f"Dataset saved to {x_file_path}")