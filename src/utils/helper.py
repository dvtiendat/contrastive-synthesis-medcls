import torch 
import random
import numpy as np

def set_seed(seed=42):
    """Set random seed for reproducibility across numpy, random, and torch.

    Args:
        seed (int): Seed value to set for reproduction. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    print(1)

if __name__ == "__main__":
    set_seed(42)
    print("Seed set to 42")

