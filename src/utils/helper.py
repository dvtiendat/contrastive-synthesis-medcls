import random
import numpy as np
import torch
import os
from pathlib import Path

def get_path():
    """Return the absolute path (root path) of the repository
    
    """
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_path, '..', '..'))
    
    return project_root
def set_seed(seed=42):
    """Set random seed for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed}")

def has_batchnorms(model):
    """Check if model has BatchNorm layers."""
    bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm)
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth', best_filename='best_model.pth'):
    """Saves checkpoint, overwrites if not best, saves separately if best."""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_filepath)
        print(f" => Saved new best model to {best_filepath}")
    else:
         print(f" => Saved checkpoint to {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scaler=None, device='cuda'):
    """Loads checkpoint state."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"=> No checkpoint found at '{checkpoint_path}'")

    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    start_epoch = checkpoint.get('epoch', 0)
    best_metric = checkpoint.get('best_metric', float('inf'))

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"=> Loaded model state dict from epoch {start_epoch}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> Loaded optimizer state dict")

    if scaler is not None and 'fp16_scaler' in checkpoint and checkpoint['fp16_scaler'] is not None:
        scaler.load_state_dict(checkpoint['fp16_scaler'])
        print("=> Loaded fp16_scaler state dict")
    elif scaler is not None and 'fp16_scaler' not in checkpoint:
        print("=> fp16_scaler state dict not found in checkpoint, scaler not loaded.")

    print(f"=> Checkpoint loaded successfully. Resuming from epoch {start_epoch}")

    return start_epoch, best_metric

def load_dino_checkpoint_for_finetune(checkpoint_path, model_backbone, device='cuda'):
    """Loads DINO teacher weights for the backbone specifically."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"=> No checkpoint found at '{checkpoint_path}'")

    print(f"Loading DINO checkpoint for finetuning from '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # DINO saves student and teacher, usually finetune from teacher
    if 'teacher' not in checkpoint:
        raise KeyError("Teacher state dict not found in DINO checkpoint.")

    # Weights are often saved under 'backbone.' prefix in MultiCropWrapper
    teacher_weights = checkpoint['teacher']
    adjusted_weights = {}
    loaded_count = 0
    skipped_count = 0
    for key, value in teacher_weights.items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '')
            adjusted_weights[new_key] = value
            loaded_count += 1
        else:
            skipped_count +=1

    if not adjusted_weights:
         raise ValueError("No weights with 'backbone.' prefix found in teacher state_dict.")

    print(f"Adjusted {loaded_count} backbone weights, skipped {skipped_count} head weights.")

    msg = model_backbone.load_state_dict(adjusted_weights, strict=False)
    print(f"=> Loaded teacher backbone weights with msg: {msg}")

    return model_backbone


def save_stats(stats, output_dir, filename='training_stats.json'):
    """Saves training statistics to a JSON file."""
    filepath = Path(output_dir) / filename
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Training stats saved to {filepath}")

