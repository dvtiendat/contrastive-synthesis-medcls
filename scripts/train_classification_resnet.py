import yaml
from src.classification.classifier_train import train_classifier

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config("configs/classification_config.yaml")
    train_classifier(cfg)
