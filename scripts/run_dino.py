import argparse
import yaml
import os
import time
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import wandb 
import timm
from pathlib import Path
from functools import partial

from src.utils.helper import *
from src.data.datasets import UnlabeledDataset
from src.data.transforms import DataAugmentationDINO
from src.models.vit import vit_small

from src.models.dino_head import DINOHead
from src.models.dino_wrapper import MultiCropWrapper
from src.contrastive.dino_losses import DINOLoss
from src.contrastive.utils import cosine_scheduler
from src.contrastive.dino_train import train_dino_epoch

from src.utils.logging import setup_logging
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='DINO Pretraining Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    parser.add_argument('--init_with_imagenet', type=lambda x: (str(x).lower() == 'true'), default=None,
                        help='Override init_with_imagenet (True/False)')
    parser.add_argument('--data_path', default=None, type=str, help='Override data path')
    parser.add_argument('--output_dir', default=None, type=str, help='Override output directory')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Override learning rate')
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.data_path is not None:
        cfg['data_path'] = args.data_path
    if args.output_dir is not None:
        cfg['output_dir'] = args.output_dir
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        cfg['learning_rate'] = args.learning_rate
    if args.init_with_imagenet is not None: 
        cfg['init_with_imagenet'] = args.init_with_imagenet
    
    print("--- Configuration ---")
    print(yaml.dump(cfg, indent=4))
    print("---------------------")

    # --- Setup ---
    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'pretrain_log.txt'
    setup_logging(log_file=str(log_file))

    # --- W&B Setup ---
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
    transform = DataAugmentationDINO(
        cfg['global_crops_scale'],
        cfg['local_crops_scale'],
        cfg['local_crops_number']
    )
    dataset = UnlabeledDataset(cfg['data_path'], transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    niter_per_ep = len(data_loader)
    logging.info(f"Data loaded: {len(dataset)} images, {niter_per_ep} iterations per epoch.")

    # --- Building student and teacher networks ---
    logging.info(f"Building student and teacher networks ({cfg['arch']})...")

    if cfg.get('init_with_imagenet', False):
        logging.info("Initializing with teacher and student backbones with ImageNet weights.")
        student_backbone_custom = vit_small(
            patch_size=cfg['patch_size'],
            embed_dim=cfg.get('embed_dim', 384),
            depth=cfg.get('depth', 12),
            num_heads=cfg.get('num_heads', 6),
            drop_path_rate=cfg.get('drop_path_rate', 0.1),
            img_size=[224]
        )
        teacher_backbone_custom = vit_small(
            patch_size=cfg['patch_size'],
            embed_dim=cfg.get('embed_dim', 384),
            depth=cfg.get('depth', 12),
            num_heads=cfg.get('num_heads', 6),
            img_size=[224]
        )
        timm_model_name = cfg['arch']
        timm_model_for_weights = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
        timm_state_dict = timm_model_for_weights.state_dict()
        msg_student = student_backbone_custom.load_state_dict(timm_state_dict, strict=False)
        logging.info(f"Loaded ImageNet weights into custom student backbone. Message: {msg_student}")
        if msg_student.missing_keys or msg_student.unexpected_keys:
            logging.warning("Potential mismatches when loading student weights. Review keys.")
            logging.warning(f"Missing: {msg_student.missing_keys}")
            logging.warning(f"Unexpected: {msg_student.unexpected_keys}")

        msg_teacher = teacher_backbone_custom.load_state_dict(timm_state_dict, strict=False)
        logging.info(f"Loaded ImageNet weights into custom teacher backbone. Message: {msg_teacher}")
        if msg_teacher.missing_keys or msg_teacher.unexpected_keys:
            logging.warning("Potential mismatches when loading teacher weights. Review keys.")
            logging.warning(f"Missing: {msg_teacher.missing_keys}")
            logging.warning(f"Unexpected: {msg_teacher.unexpected_keys}")


        student_backbone = student_backbone_custom
        teacher_backbone = teacher_backbone_custom

    else:
        logging.info("Initializing student and teacher backbones from scratch.")
    # Use partial for norm_layer if needed
        student_backbone = vit_small(patch_size=cfg['patch_size'], drop_path_rate=cfg.get('drop_path_rate', 0.1))
        teacher_backbone = vit_small(patch_size=cfg['patch_size'], drop_path_rate=0.0) 

    embed_dim = student_backbone.embed_dim

    student = MultiCropWrapper(
        student_backbone,
        DINOHead(embed_dim, cfg['out_dim'], use_bn=cfg['use_bn_in_head'], norm_last_layer=cfg['norm_last_layer'])
    )
    teacher = MultiCropWrapper(
        teacher_backbone,
        DINOHead(embed_dim, cfg['out_dim'], use_bn=cfg['use_bn_in_head'], norm_last_layer=cfg['norm_last_layer'])
    )
    student.to(device)
    teacher.to(device)

    # Initialize teacher with student weights and freeze teacher
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    logging.info("Student and Teacher models built. Teacher parameters frozen.")

    # --- DINO Loss ---
    dino_loss = DINOLoss(
        cfg['out_dim'],
        cfg['local_crops_number'] + 2, # total number of crops = local_crops + 2 global
        cfg['warmup_teacher_temp'],
        cfg['teacher_temp'],
        cfg['warmup_teacher_temp_epochs'],
        cfg['epochs'],
        cfg['student_temp'],
        cfg['center_momentum']
    ).to(device)

    # --- Optimizer ---
    # Separate parameters for different weight decay
    params_groups = []
    for n, p in student.named_parameters():
        if not p.requires_grad:
            continue
        # Apply no weight decay to bias and norm layers
        if "bias" in n or "norm" in n:
             params_groups.append({"params": [p], "weight_decay": 0.})
        else:
             params_groups.append({"params": [p]})

    # Scale learning rate
    base_lr = cfg['learning_rate']
    scaled_lr = base_lr * cfg['batch_size'] / 256.0 # Linear scaling rule

    optimizer = torch.optim.AdamW(params_groups, lr=scaled_lr, weight_decay=cfg['weight_decay']) # WD is applied group-wise
    logging.info(f"Optimizer AdamW created. Scaled LR: {scaled_lr:.6f}")

    # --- FP16 Scaler ---
    fp16_scaler = None
    if cfg['use_fp16']:
        fp16_scaler = torch.cuda.amp.GradScaler()
        logging.info("Using FP16 training.")

    # --- Schedulers ---
    lr_schedule = cosine_scheduler(
        scaled_lr, cfg['min_lr'], cfg['epochs'], niter_per_ep, warmup_epochs=cfg['warmup_epochs']
    )
    wd_schedule = cosine_scheduler(
        cfg['weight_decay'], cfg['weight_decay_end'], cfg['epochs'], niter_per_ep
    )
    momentum_schedule = cosine_scheduler(
        cfg['momentum_teacher'], 1.0, cfg['epochs'], niter_per_ep
    )
    logging.info("LR, WD, and Momentum schedules created.")

    start_epoch = 0

    # --- Training Loop ---
    logging.info("Starting DINO pretraining!")
    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(start_epoch, cfg['epochs']):
        cfg['current_epoch'] = epoch # Add epoch to config for logging inside train_step

        train_stats = train_dino_epoch(
            student, teacher, dino_loss, data_loader, optimizer,
            lr_schedule, wd_schedule, momentum_schedule, epoch, cfg['epochs'], fp16_scaler, cfg
        )

        # --- Save Checkpoint ---
        is_best = train_stats['loss'] < best_loss
        best_loss = min(train_stats['loss'], best_loss)

        save_dict = {
            'epoch': epoch + 1,
            'arch': cfg['arch'],
            'patch_size': cfg['patch_size'],
            'out_dim': cfg['out_dim'],
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': train_stats['loss'],
            'dino_loss': dino_loss.state_dict(),
            'config': cfg # Save config in checkpoint
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()

        save_checkpoint(save_dict, is_best, str(output_dir)) 
        if cfg['saveckp_freq'] > 0 and (epoch + 1) % cfg['saveckp_freq'] == 0:
             save_checkpoint(save_dict, False, str(output_dir), filename=f'checkpoint{epoch:04}.pth') 

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        if cfg.get('use_wandb', True):
            wandb.log({"epoch_loss": train_stats['loss'], "epoch": epoch})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Pretraining finished. Total time: {total_time_str}')
    if cfg.get('use_wandb', True):
        wandb.finish()

if __name__ == '__main__':
    main()