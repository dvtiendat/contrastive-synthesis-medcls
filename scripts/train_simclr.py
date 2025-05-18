# run_simclr.py

import yaml
from src.contrastive.simclr_train import train_simclr

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config("configs/contrastive_config.yaml")
    train_simclr(cfg)
