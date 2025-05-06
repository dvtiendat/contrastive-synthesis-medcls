import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import wandb

from torch.cuda.amp import GradScaler, autocast

def train_step(model, dataloader, loss_fn, optimizer, device, num_classes, cfg, scaler): # <<<--- ADD scaler argument
    """Performs a single training step/epoch for classification with FP16 support."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    use_fp16 = cfg.get('use_fp16', False)

    for batch, (images, labels) in enumerate(dataloader):
        if images is None or labels is None:
            print(f"Skipping batch {batch} due to invalid samples.")
            continue
        images, labels = images.to(device), labels.to(device)

        # --- FP16: Wrap forward pass with autocast ---
        with autocast(enabled=use_fp16):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()

        # --- FP16: Use scaler for backward and step ---
        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: # Standard backward/step if FP16 is disabled or scaler is None
            loss.backward()
            optimizer.step()

        # Metrics calculation
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_train_loss = train_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0

    # Calculate other metrics (remains the same)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_acc = {}
    for i in range(num_classes):
        class_mask = (all_labels == i)
        class_total = np.sum(class_mask)
        per_class_acc[i] = 100 * np.sum((all_preds == i) & class_mask) / class_total if class_total > 0 else 0

    metrics = { # metrics dictionary remains the same
        'loss': avg_train_loss, 'accuracy': accuracy, 'f1_macro': f1_macro,
        'f1_weighted': f1_weighted, 'per_class_acc': per_class_acc,
        'f1_per_class': f1_per_class, 'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class
    }

    # W&B Logging (remains the same)
    if cfg.get('use_wandb', True) and wandb.run:
        wandb.log({
            "train_loss": metrics['loss'], "train_accuracy": metrics['accuracy'],
            "train_f1_macro": metrics['f1_macro'], "train_f1_weighted": metrics['f1_weighted']
        }, commit=False)

    return metrics


def val_step(model, dataloader, loss_fn, device, num_classes, cfg):
    """Performs a single validation step/epoch for classification with FP16 support."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    use_fp16 = cfg.get('use_fp16', False) 

    with torch.inference_mode():
        with autocast(enabled=use_fp16): 
            for batch, (images, labels) in enumerate(dataloader):
                if images is None or labels is None:
                     print(f"Skipping validation batch {batch} due to invalid samples.")
                     continue
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    per_class_acc = {}
    for i in range(num_classes):
        class_mask = (all_labels == i)
        class_total = np.sum(class_mask)
        per_class_acc[i] = 100 * np.sum((all_preds == i) & class_mask) / class_total if class_total > 0 else 0

    metrics = { 
        'loss': avg_val_loss, 'accuracy': accuracy, 'f1_macro': f1_macro,
        'f1_weighted': f1_weighted, 'per_class_acc': per_class_acc,
        'f1_per_class': f1_per_class, 'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class, 'all_preds': all_preds,
        'all_labels': all_labels, 'confusion_matrix': conf_matrix
    }

    if cfg.get('use_wandb', True) and wandb.run:
        log_data = {
            "val_loss": metrics['loss'], "val_accuracy": metrics['accuracy'],
            "val_f1_macro": metrics['f1_macro'], "val_f1_weighted": metrics['f1_weighted'],
            "epoch": cfg.get('current_epoch', -1) 
        }
        wandb.log(log_data, commit=True)

    return metrics