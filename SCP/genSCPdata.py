import numpy as np
import random

def generate_scp_dataset(num_universe_items, num_sets, min_set_size, max_set_size):
    """
    Generate a dataset for the Set Covering Problem.

    :param num_universe_items: Number of items in the universe.
    :param num_sets: Number of sets to generate.
    :param min_set_size: Minimum number of items in each set.
    :param max_set_size: Maximum number of items in each set.
    :return: A tuple (universe, sets) where universe is a set of all items,
             and sets is a list of sets, each containing a subset of the items.
    """
    # Generate the universe of items
    universe = set(range(num_universe_items))

    # Generate the sets
    sets = []
    for _ in range(num_sets):
        set_size = random.randint(min_set_size, max_set_size)
        sets.append(set(random.sample(universe, set_size)))

    # Ensure every item is covered at least once
    uncovered_items = universe.copy()
    for item in universe:
        if not any(item in s for s in sets):
            sets[random.randint(0, num_sets-1)].add(item)
            uncovered_items.discard(item)

    # If there are still uncovered items, add them to random sets
    while uncovered_items:
        item = uncovered_items.pop()
        sets[random.randint(0, num_sets-1)].add(item)

    return universe, sets

# Usage
num_universe_items = 50  # Total number of items in the universe
num_sets = 20            # Total number of sets
min_set_size = 3         # Minimum number of items in each set
max_set_size = 10        # Maximum number of items in each set

universe, sets = generate_scp_dataset(num_universe_items, num_sets, min_set_size, max_set_size)

# Output the generated instance
print("Universe:", universe)
print("Sets:")
for s in sets:
    print(s)
