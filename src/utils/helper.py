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
    abs_path = os.path.abspath(checkpoint_path) # Get absolute path for clarity
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"=> No checkpoint found at '{abs_path}'")

    print(f"\n--- Loading DINO Checkpoint for Finetune ---")
    print(f"=> Attempting to load checkpoint from: '{abs_path}'")
    try:
        # Try loading with weights_only=False first, as it worked in notebook
        # and might be necessary if non-tensor data is pickled
        checkpoint = torch.load(abs_path, map_location=device, weights_only=False)
        print("=> Checkpoint loaded successfully (weights_only=False).")
    except Exception as e_false:
        print(f"=> Loading with weights_only=False failed: {e_false}")
        try:
            # Fallback to weights_only=True if False failed unexpectedly
            # (This might fail again if non-tensor data is present)
            print("=> Attempting to load with weights_only=True...")
            checkpoint = torch.load(abs_path, map_location=device, weights_only=True)
            print("=> Checkpoint loaded successfully (weights_only=True).")
        except Exception as e_true:
            print(f"=> Loading with weights_only=True also failed: {e_true}")
            print("=> ERROR: Could not load the checkpoint file.")
            raise e_true # Re-raise the error

    # --- Check Keys ---
    print(f"=> Checkpoint keys: {list(checkpoint.keys())}")
    if 'teacher' not in checkpoint:
        # Maybe the key is different? Or student was saved?
        if 'state_dict' in checkpoint: # Common alternative
             print("WARNING: 'teacher' key not found, trying 'state_dict'.")
             teacher_weights = checkpoint['state_dict']
        elif 'student' in checkpoint:
             print("WARNING: 'teacher' key not found, trying 'student'. Fine-tuning might be suboptimal.")
             teacher_weights = checkpoint['student']
        else:
             raise KeyError("Neither 'teacher' nor 'state_dict' nor 'student' key found in DINO checkpoint.")
    else:
        teacher_weights = checkpoint['teacher']
        print(f"=> Found 'teacher' state dict with {len(teacher_weights)} keys.")


    # --- Key Stripping and Loading ---
    adjusted_weights = {}
    loaded_count = 0
    skipped_count = 0
    print("\n--- Processing Teacher Weights ---")
    for key, value in teacher_weights.items():
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', '', 1) # Replace only the first occurrence
            adjusted_weights[new_key] = value
            loaded_count += 1
            # print(f"Mapping: '{key}' -> '{new_key}'") # Uncomment for very detailed debug
        else:
            skipped_count +=1
            # print(f"Skipping non-backbone key: {key}") # Uncomment for very detailed debug

    if loaded_count == 0:
         print("\nERROR: No weights with 'backbone.' prefix found in the selected state_dict.")
         print("Available keys were:")
         # Print first 10 keys from the loaded state_dict for inspection
         keys_to_show = list(teacher_weights.keys())[:10]
         for k in keys_to_show: print(f"  - {k}")
         if len(teacher_weights) > 10: print("  ...")
         raise ValueError("Could not extract backbone weights. Check checkpoint structure and prefixes.")
    else:
        print(f"Processed {loaded_count} backbone weights, skipped {skipped_count} other weights (likely head).")

    # --- Load into Backbone ---
    print("\n--- Loading State Dict into Backbone ---")
    # Set strict=True first to catch all mismatches. If it fails, try strict=False
    # but investigate the mismatches printed.
    try:
        msg = model_backbone.load_state_dict(adjusted_weights, strict=True)
        print("=> Loaded teacher backbone weights successfully (strict=True).")
        if msg.missing_keys: print(f"Missing keys in model: {msg.missing_keys}") # Should be empty
        if msg.unexpected_keys: print(f"Unexpected keys in checkpoint: {msg.unexpected_keys}") # Should be empty
    except RuntimeError as e_strict:
        print(f"\nWARNING: Loading with strict=True failed: {e_strict}")
        print("=> Attempting to load with strict=False...")
        # Load non-strictly to see what keys are mismatched
        msg = model_backbone.load_state_dict(adjusted_weights, strict=False)
        print("=> Loaded teacher backbone weights potentially incompletely (strict=False).")
        if msg.missing_keys:
             print(f"WARNING: Missing keys in model state_dict: {msg.missing_keys}")
             print("         (These layers in your finetune model did not get weights from the checkpoint)")
        if msg.unexpected_keys:
             print(f"WARNING: Unexpected keys in checkpoint state_dict: {msg.unexpected_keys}")
             print("         (These weights from the checkpoint were not used by your finetune model)")
        if not msg.missing_keys and not msg.unexpected_keys:
             print("         (strict=False loading reported no mismatches, error might be subtle)")

    print("------------------------------------------\n")
    return model_backbone


def save_stats(stats, output_dir, filename='training_stats.json'):
    """Saves training statistics to a JSON file."""
    filepath = Path(output_dir) / filename
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Training stats saved to {filepath}")

