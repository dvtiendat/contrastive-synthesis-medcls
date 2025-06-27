import argparse
import yaml
import os
import random
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from functools import partial

from src.utils.helper import set_seed
from src.data.datasets import Classification_dataset, collate_fn
from src.data.transforms import get_val_transform 
from src.models.vit import vit_small 
from src.models.vit_cls_head import FinetuneViT
from src.classification.train_eval import val_step
from src.utils.logging import setup_logging
import logging

def plot_confusion_matrix(cm, classes, filename):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(filename)
    plt.close()
    logging.info(f"Confusion matrix saved to {filename}")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Fine-tuned ViT Model on Test Set')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the FINE-TUNING config YAML file used during training.')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the best model checkpoint file (e.g., best_finetune_model.pth).')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override data path if different from config.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results (e.g., confusion matrix plot). Defaults to checkpoint dir.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size for evaluation (optional).')
    parser.add_argument('--device', type=str, default=None, help='Device override (e.g., cpu, cuda:0)')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    print("--- Using Configuration from Training ---")
    print(yaml.dump(cfg, indent=4))
    print("----------------------------------------")

    if args.data_path is not None: cfg['data_path'] = args.data_path
    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.get('seed', 42))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    log_file = output_dir / 'evaluate_log.txt'
    setup_logging(log_file=str(log_file))
    logging.info(f"Using device: {device}")
    logging.info(f"Evaluating checkpoint: {args.checkpoint_path}")

    cfg['use_wandb'] = False

    logging.info("Preparing test dataset...")
    val_transform = get_val_transform(cfg['img_size'])
    classes = ['COVID', 'Lung_Opacity', 'Viral_Pneumonia', 'Normal'] 

    dataset_full_for_split = Classification_dataset(cfg['data_path'], transform=None, classes=classes)
    all_indices = list(range(len(dataset_full_for_split)))

    random.seed(cfg.get('seed', 42))
    random.shuffle(all_indices)
    train_size = int(cfg['train_split'] * len(all_indices))
    val_size = int(cfg['val_split'] * len(all_indices))
    test_indices = all_indices[train_size + val_size:]

    test_dataset = Subset(
        Classification_dataset(cfg['data_path'], transform=val_transform, classes=classes),
        test_indices
    )

    test_loader = DataLoader(test_dataset, batch_size=cfg.get('batch_size', 32), shuffle=False, # No shuffle for test
                             num_workers=cfg.get('num_workers', 4), pin_memory=True, collate_fn=collate_fn)

    logging.info(f"Test set size: {len(test_dataset)}")
    if len(test_dataset) == 0:
        logging.error("Test dataset is empty! Check data path and split logic.")
        return

    # --- Build Model ---
    logging.info(f"Building model architecture ({cfg['arch']})...")
    backbone = vit_small()
    # Instantiate the full model
    model = FinetuneViT(
        backbone,
        num_classes=cfg['num_classes'],
        freeze_backbone='False'
    )
    model.to(device)

    # --- Load Best Checkpoint Weights ---
    logging.info(f"Loading weights from: {args.checkpoint_path}")
    if not os.path.isfile(args.checkpoint_path):
        logging.error(f"Checkpoint file not found at {args.checkpoint_path}")
        return
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Checkpoint does not contain 'model_state_dict'.")
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Successfully loaded model state dict from epoch {checkpoint.get('epoch', 'N/A')}")
    except Exception as e:
        logging.error(f"Failed to load model weights: {e}")
        return

    # --- Evaluate on Test Set ---
    logging.info("Starting evaluation on the test set...")
    criterion = nn.CrossEntropyLoss()
    model.eval()

    test_metrics = val_step(model, test_loader, criterion, device, cfg['num_classes'], cfg)

    logging.info("\n--- Test Set Results ---")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}%")
    print(f"F1 Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {test_metrics['f1_weighted']:.4f}")

    print("\n--- Per-Class Metrics ---")
    num_classes = cfg['num_classes']
    classes = ['COVID', 'Lung_Opacity', 'Viral_Pneumonia', 'Normal'] 
    print("Precision:")
    for i in range(num_classes):
        print(f"  Class {i} ({classes[i]}): {test_metrics['precision_per_class'][i]:.4f}")

    print("Recall:")
    for i in range(num_classes):
        print(f"  Class {i} ({classes[i]}): {test_metrics['recall_per_class'][i]:.4f}")

    print("F1 Score:")
    for i in range(num_classes):
        print(f"  Class {i} ({classes[i]}): {test_metrics['f1_per_class'][i]:.4f}")

    print("Accuracy:")
    for i in range(num_classes):
        acc_val = test_metrics['per_class_acc'].get(i, 'N/A')
        if isinstance(acc_val, float):
            print(f"  Class {i} ({classes[i]}): {acc_val:.2f}%")
        else:
            print(f"  Class {i} ({classes[i]}): {acc_val}")
    print("--------------------------")

    # Detailed report using classification_report
    print("\nClassification Result:")
    report = classification_report(
        test_metrics['all_labels'],
        test_metrics['all_preds'],
        target_names=classes,
        digits=4,
        zero_division=0
    )
    print(report)
    logging.info("\nClassification Report:\n" + report)

    # Save and display confusion matrix
    cm_filename = output_dir / "test_confusion_matrix.png"
    plot_confusion_matrix(test_metrics['confusion_matrix'], classes, str(cm_filename))
    print(f"Confusion matrix plot saved to {cm_filename}")

    logging.info("Evaluation finished.")

if __name__ == '__main__':
    main()