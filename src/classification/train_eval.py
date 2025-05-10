import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import wandb # Import W&B

def train_step(model, dataloader, loss_fn, optimizer, device, num_classes, cfg):
    """Performs a single training step/epoch for classification."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch, (images, labels) in enumerate(dataloader):
        if images is None or labels is None:
            print(f"Skipping batch {batch} due to invalid samples.")
            continue
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
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

    # Calculate other metrics
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
        if class_total > 0:
            per_class_acc[i] = 100 * np.sum((all_preds == i) & class_mask) / class_total
        else:
            per_class_acc[i] = 0

    metrics = {
        'loss': avg_train_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class_acc': per_class_acc,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class
    }

    # Log to W&B if enabled
    if cfg.get('use_wandb', True):
        wandb.log({
            "train_loss": metrics['loss'],
            "train_accuracy": metrics['accuracy'],
            "train_f1_macro": metrics['f1_macro'],
            "train_f1_weighted": metrics['f1_weighted']
            # Add per-class metrics if desired, prefixing with 'train_'
        }, commit=False) # Commit=False because val_step will commit

    return metrics


def val_step(model, dataloader, loss_fn, device, num_classes, cfg):
    """Performs a single validation step/epoch for classification."""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
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

    # Calculate metrics
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
        if class_total > 0:
            per_class_acc[i] = 100 * np.sum((all_preds == i) & class_mask) / class_total
        else:
            per_class_acc[i] = 0

    metrics = {
        'loss': avg_val_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'per_class_acc': per_class_acc,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'all_preds': all_preds, # Keep for potential analysis
        'all_labels': all_labels, # Keep for potential analysis
        'confusion_matrix': conf_matrix # Return confusion matrix
    }

    # Log to W&B if enabled
    if cfg.get('use_wandb', True):
        log_data = {
            "val_loss": metrics['loss'],
            "val_accuracy": metrics['accuracy'],
            "val_f1_macro": metrics['f1_macro'],
            "val_f1_weighted": metrics['f1_weighted'],
            "epoch": cfg['current_epoch'] # Add epoch for proper step tracking
            # Add per-class metrics if desired, prefixing with 'val_'
        }
        # Log confusion matrix (optional, requires wandb.plot)
        # class_names = dataloader.dataset.dataset.classes # Get class names carefully
        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(preds=all_preds, y_true=all_labels, class_names=class_names)})
        wandb.log(log_data, commit=True) # Commit True after logging both train and val metrics

    return metrics