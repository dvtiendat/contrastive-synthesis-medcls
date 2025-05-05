import argparse
import yaml
import os
import time
import datetime
import json
import random
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb 
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.helper import set_seed
from src.data.datasets import Classification_dataset, collate_fn
from src.data.transforms import get_train_transform, get_val_transform
from src.models.vit import vit_small 
from src.models.vit_cls_head import FinetuneViT
from src.classification.train_eval import train_step, val_step
from src.utils.helper import save_checkpoint, load_checkpoint, load_dino_checkpoint_for_finetune, save_stats
from src.utils.logging import setup_logging
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='ViT Finetuning Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    parser.add_argument('--data_path', type=str, default=None, help='Override data path for labeled data')
    parser.add_argument('--output_dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--pretrained_checkpoint_path', type=str, default=None, help='Override path to pretrained DINO checkpoint')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--freeze_backbone', type=bool, default=None, help='Override whether to freeze backbone')
    
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, filename):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.close() 
    logging.info(f"Confusion matrix saved to {filename}")
    return wandb.Image(filename) if wandb.run else None 

def main():
    args = parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override config values with command line arguments if provided
    if args.data_path is not None:
        cfg['data_path'] = args.data_path
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.pretrained_checkpoint_path is not None:
        cfg['pretrained_checkpoint_path'] = args.pretrained_checkpoint_path
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        cfg['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.freeze_backbone is not None:
        cfg['freeze_backbone'] = args.freeze_backbone
    
    print("--- Configuration ---")
    print(yaml.dump(cfg, indent=4))
    print("---------------------")

    # --- Setup ---
    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'finetune_log.txt'
    setup_logging(log_file=str(log_file))

    if cfg.get('use_wandb', True):
        try:
            wandb.init(
                project=cfg['wandb_project'],
                entity=cfg.get('wandb_entity'),
                config=cfg,
                name=cfg['wandb_run_name']
            )
            print("Weights & Biases initialized.")
        except Exception as e:
            print(f"Could not initialize W&B: {e}. Continuing without W&B.")
            cfg['use_wandb'] = False
    else:
         print("W&B logging disabled by config.")

    # --- Prepare Data ---
    logging.info("Preparing dataset...")
    train_transform = get_train_transform(cfg['img_size'])
    val_transform = get_val_transform(cfg['img_size'])

    classes = ['COVID', 'Lung_Opacity', 'Viral_Pneumonia', 'Normal']
    dataset_full_train = Classification_dataset(cfg['data_path'], transform=train_transform, classes=classes)
    dataset_full_val = Classification_dataset(cfg['data_path'], transform=val_transform, classes=classes)

    all_indices = list(range(len(dataset_full_train)))
    random.shuffle(all_indices) 
    train_size = int(cfg['train_split'] * len(all_indices))
    val_size = int(cfg['val_split'] * len(all_indices))
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]

    train_dataset = Subset(dataset_full_train, train_indices)
    val_dataset = Subset(dataset_full_val, val_indices)
    test_dataset = Subset(dataset_full_val, test_indices) 

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=cfg['num_workers'], pin_memory=True, collate_fn=collate_fn)

    logging.info(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_indices)}")


    # --- Building Model ---
    logging.info(f"Building finetuning model ({cfg['arch']})...")
    # Instantiate the backbone first
    backbone = vit_small(patch_size=cfg['patch_size'], img_size=[cfg['img_size']]) 

    # Load pretrained DINO weights into the backbone
    if cfg.get('pretrained_checkpoint_path'):
        try:
             backbone = load_dino_checkpoint_for_finetune(cfg['pretrained_checkpoint_path'], backbone, device=device)
             logging.info("Loaded DINO pretrained weights into backbone.")
        except Exception as e:
             logging.error(f"Failed to load DINO checkpoint: {e}.")
             # Optionally load ImageNet weights here if DINO fails and specified
             backbone = timm.create_model(cfg['arch'], pretrained=True, num_classes=0)
    else:
        logging.warning("No pretrained_checkpoint_path specified.")
        # Or load ImageNet weights by default if desired
        backbone = timm.create_model(cfg['arch'], pretrained=True, num_classes=0)


    model = FinetuneViT(backbone, num_classes=cfg['num_classes'], freeze_backbone=cfg['freeze_backbone'])
    model.to(device)
    if cfg.get('use_wandb', True):
        wandb.watch(model, log_freq=100)

    # --- Loss Function and Optimizer ---
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) 
    logging.info(f"Optimizer AdamW and ReduceLROnPlateau scheduler created. LR: {cfg['learning_rate']:.6f}")

    start_epoch = 0
    best_val_loss = float('inf')

    # --- Training Loop ---
    logging.info("Starting finetuning!")
    start_time = time.time()
    stats = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'f1_macro': [], 'f1_weighted': []}
    epochs_no_improve = 0

    for epoch in range(start_epoch, cfg['epochs']):
        cfg['current_epoch'] = epoch # Add epoch to config for logging inside steps
        epoch_start_time = time.time()

        train_metrics = train_step(model, train_loader, criterion, optimizer, device, cfg['num_classes'], cfg)
        val_metrics = val_step(model, val_loader, criterion, device, cfg['num_classes'], cfg)

        scheduler.step(val_metrics['loss'])

        stats['train_loss'].append(train_metrics['loss'])
        stats['val_loss'].append(val_metrics['loss'])
        stats['train_acc'].append(train_metrics['accuracy'])
        stats['val_acc'].append(val_metrics['accuracy'])
        stats['f1_macro'].append(val_metrics['f1_macro'])
        stats['f1_weighted'].append(val_metrics['f1_weighted'])

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch [{epoch+1}/{cfg['epochs']}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.2f}% | Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val F1 Macro: {val_metrics['f1_macro']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
             best_val_loss = val_metrics['loss']
             epochs_no_improve = 0 
             save_dict = {
                 'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'best_val_loss': best_val_loss,
                 'val_acc': val_metrics['accuracy'],
                 'f1_macro': val_metrics['f1_macro'],
                 'config': cfg
             }
             save_checkpoint(save_dict, is_best, str(output_dir), best_filename='best_finetune_model.pth')
        else:
             epochs_no_improve += 1

        if epochs_no_improve >= cfg.get('early_stop_patience', 10):
             logging.info(f"Early stopping triggered at epoch {epoch+1} after {epochs_no_improve} epochs without improvement.")
             break

    final_save_dict = {
        'epoch': epoch + 1, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': val_metrics['loss'],
        'final_val_acc': val_metrics['accuracy'],
        'final_f1_macro': val_metrics['f1_macro'],
        'config': cfg
    }
    save_checkpoint(final_save_dict, False, str(output_dir), filename='last_finetune_model.pth')

    cm_filename = output_dir / f"confusion_matrix_epoch_{epoch+1}.png"
    cm_image = plot_confusion_matrix(val_metrics['confusion_matrix'], classes, str(cm_filename))
    if cfg.get('use_wandb', True) and cm_image:
         wandb.log({"final_confusion_matrix": cm_image, "epoch": epoch}) # Log with epoch


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Finetuning finished. Total time: {total_time_str}')
    save_stats(stats, output_dir, 'finetuning_stats.json')

    if cfg.get('use_wandb', True):
        wandb.finish()


if __name__ == '__main__':
    main()