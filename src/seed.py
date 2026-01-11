import random
import numpy as np

def init_seed(seed):
    if seed is None:
        seed = 42 # Default seed
    random.seed(seed)
    np.random.seed(seed)