import torch
import time
import numpy as np
import wandb 
from .utils import clip_gradients, cancel_gradients_last_layer 

def train_dino_epoch(student, teacher, dino_loss, data_loader,
                     optimizer, lr_schedule, wd_schedule, momentum_schedule,
                     epoch, total_epochs, fp16_scaler, cfg):
    """ Trains DINO for one epoch. """
    student.train()
    teacher.eval() # Teacher is always in eval mode

    total_loss = 0.0
    start_time = time.time()

    # Use niter_per_ep for scheduler indexing
    niter_per_ep = len(data_loader)

    for it, (images, _) in enumerate(data_loader):
        # Calculate global iteration number
        global_it = epoch * niter_per_ep + it

        # Update learning rate and weight decay based on schedules
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if i == 0:  # Only the first group (usually non-bias/norm) gets full WD
                param_group["weight_decay"] = wd_schedule[global_it]

        # Move images to GPU
        images = [im.cuda(non_blocking=True) for im in images]

        # Forward pass with Automatic Mixed Precision (AMP) if enabled
        with torch.cuda.amp.autocast(enabled=(fp16_scaler is not None)):
            # Teacher forward pass (only global crops, no gradients)
            with torch.no_grad():
                 teacher_output = teacher(images[:2]) # First 2 are global crops

            # Student forward pass (all crops)
            student_output = student(images)

            # Calculate DINO loss
            loss = dino_loss(student_output, teacher_output, epoch)

        # --- Backward pass and optimization ---
        optimizer.zero_grad()

        if fp16_scaler is None: # No FP16
            loss.backward()
            if cfg['clip_grad'] > 0:
                _ = clip_gradients(student, cfg['clip_grad']) # Use returned norms?
            cancel_gradients_last_layer(epoch, student, cfg['freeze_last_layer'])
            optimizer.step()
        else: # With FP16
            fp16_scaler.scale(loss).backward()
            if cfg['clip_grad'] > 0:
                fp16_scaler.unscale_(optimizer) # Unscale before clipping
                _ = clip_gradients(student, cfg['clip_grad'])
            cancel_gradients_last_layer(epoch, student, cfg['freeze_last_layer'])
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # --- Teacher momentum update (EMA) ---
        with torch.no_grad():
            m = momentum_schedule[global_it] # Current momentum value
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging
        batch_loss = loss.item()
        total_loss += batch_loss
        if global_it % 50 == 0: # Log every 50 iterations
            print(f"Epoch [{epoch+1}/{total_epochs}] Iter [{it+1}/{niter_per_ep}] Loss: {batch_loss:.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")
            if cfg.get('use_wandb', True): # Check if W&B is enabled in config
                wandb.log({
                    "dino_batch_loss": batch_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "weight_decay": optimizer.param_groups[0]['weight_decay'],
                    "teacher_momentum": m,
                    "global_step": global_it,
                    "epoch": epoch
                 })

    avg_epoch_loss = total_loss / niter_per_ep
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Time: {epoch_time:.2f}s")

    return {"loss": avg_epoch_loss, "lr": optimizer.param_groups[0]['lr']}