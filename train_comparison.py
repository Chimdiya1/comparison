"""
Unified training script for comparing FCN-8s, UNet, and DeepLabV3+ on building damage assessment.

This script ensures a fair comparison by using identical:
- Hyperparameters (LR, batch size, optimizer)
- Data augmentation
- Loss function (Combined CE + Dice)
- Training protocol (epochs, early stopping)
- Random seed for reproducibility

Usage:
    python assessment/train_comparison.py --model fcn8s
    python assessment/train_comparison.py --model unet
    python assessment/train_comparison.py --model deeplabv3plus
"""

import os
import sys
import csv
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add paths for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from unet_model import UNet
from fcn import FCN8s
from deeplabv3plus import DeepLabV3Plus


# ----------------------------
# Configuration
# ----------------------------
class Config:
    """Training configuration - IDENTICAL for all models"""

    # Data paths
    TRAIN_CSV = "/content/train/tiles_256/train_tiles_balanced.csv"
    VAL_CSV = "/content/train/tiles_256/val_tiles.csv"

    # Output directory
    OUT_DIR = "/content/assessment/results"

    # Model parameters
    NUM_CLASSES = 5
    IN_CHANNELS = 6

    # Training hyperparameters (IDENTICAL across all models)
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 1e-3
    WEIGHT_DECAY = 0.0
    NUM_WORKERS = 2

    # Loss weights
    CE_WEIGHT = 1.0
    DICE_WEIGHT = 1.0
    DICE_EXCLUDE_BG = True

    # Early stopping
    PATIENCE = 5

    # Reproducibility
    SEED = 42

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute class weights from training data
    COMPUTE_CLASS_WEIGHTS = True


# ----------------------------
# Data Augmentation (from existing training script)
# ----------------------------
def augment_tile(x6: np.ndarray, mask: np.ndarray, rng: random.Random):
    """
    Apply spatial and photometric augmentations.

    Spatial: horizontal/vertical flip, rotation (90°, 180°, 270°)
    Photometric: brightness, contrast adjustment
    """
    # Spatial augmentations
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=1)
        mask = np.flip(mask, axis=1)
    if rng.random() < 0.5:
        x6 = np.flip(x6, axis=0)
        mask = np.flip(mask, axis=0)
    k = rng.randint(0, 3)
    if k:
        x6 = np.rot90(x6, k, axes=(0, 1))
        mask = np.rot90(mask, k, axes=(0, 1))

    # Photometric augmentations (input only)
    if x6.dtype != np.float32 and x6.dtype != np.float64:
        x6f = x6.astype(np.float32) / 255.0
    else:
        x6f = x6.astype(np.float32)

    # Brightness
    if rng.random() < 0.8:
        x6f = x6f + rng.uniform(-0.08, 0.08)

    # Contrast
    if rng.random() < 0.8:
        c = rng.uniform(0.9, 1.1)
        mean = x6f.mean(axis=(0, 1), keepdims=True)
        x6f = (x6f - mean) * c + mean

    x6f = np.clip(x6f, 0.0, 1.0).astype(np.float32)
    return x6f, mask.astype(np.uint8)


class XBDTilesDataset(Dataset):
    """Dataset for xView2 building damage tiles"""

    def __init__(self, csv_path: str, augment: bool = False, seed: int = 42):
        self.augment = augment
        self.seed = seed

        self.rows = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        if not self.rows:
            raise ValueError(f"No rows found in {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        x6 = np.load(r["sixch_tile_path"])  # (256,256,6)
        mask = np.array(Image.open(r["mask_tile_path"]), dtype=np.uint8)  # (256,256)

        if self.augment:
            rng = random.Random(self.seed * 10_000 + idx)
            x6, mask = augment_tile(x6, mask, rng)
        else:
            if x6.dtype != np.float32 and x6.dtype != np.float64:
                x6 = x6.astype(np.float32) / 255.0
            else:
                x6 = x6.astype(np.float32)

        x_t = torch.from_numpy(x6).permute(2, 0, 1)  # (6,H,W)
        y_t = torch.from_numpy(mask.astype(np.int64))  # (H,W)
        return x_t, y_t


# ----------------------------
# Loss Functions
# ----------------------------
class SoftDiceLoss(nn.Module):
    """Soft Dice Loss for semantic segmentation"""

    def __init__(self, num_classes: int, exclude_bg: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.exclude_bg = exclude_bg
        self.eps = eps

    def forward(self, logits, target):
        # logits: (N,C,H,W), target: (N,H,W)
        probs = F.softmax(logits, dim=1)  # (N,C,H,W)
        target_1h = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.exclude_bg:
            probs = probs[:, 1:, :, :]
            target_1h = target_1h[:, 1:, :, :]

        # Flatten
        probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)  # (N,C,H*W)
        target_flat = target_1h.reshape(target_1h.shape[0], target_1h.shape[1], -1)

        intersection = (probs_flat * target_flat).sum(dim=2)  # (N,C)
        union = probs_flat.sum(dim=2) + target_flat.sum(dim=2)  # (N,C)

        dice = (2.0 * intersection + self.eps) / (union + self.eps)  # (N,C)
        loss = 1.0 - dice.mean()
        return loss


class CombinedLoss(nn.Module):
    """Combined Cross-Entropy + Dice Loss"""

    def __init__(self, num_classes: int, ce_weight: float = 1.0, dice_weight: float = 1.0,
                 class_weights: torch.Tensor = None, exclude_bg: bool = True):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = SoftDiceLoss(num_classes, exclude_bg=exclude_bg)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


# ----------------------------
# Utility Functions
# ----------------------------
def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_class_weights(dataloader, num_classes: int, device: str):
    """Compute inverse frequency class weights from training data"""
    print("Computing class weights from training data...")
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, masks in dataloader:
        for c in range(num_classes):
            counts[c] += (masks == c).sum().item()

    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = weights.to(device)

    print(f"Class counts: {counts.numpy()}")
    print(f"Class weights: {weights.cpu().numpy()}")
    return weights


def create_model(model_name: str, in_channels: int, num_classes: int, pretrained: bool = True):
    """Create model by name"""
    if model_name == "fcn8s":
        return FCN8s(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    elif model_name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes, base=32)
    elif model_name == "deeplabv3plus":
        return DeepLabV3Plus(in_channels=in_channels, num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------
# Training Loop
# ----------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

    avg_loss = total_loss / num_batches
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model_name: str, config: Config):
    """Main training function"""

    # Set seed for reproducibility
    set_seed(config.SEED)

    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()} Model")
    print(f"{'='*80}\n")

    # Create output directory
    model_dir = os.path.join(config.OUT_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Create model
    print("Creating model...")
    model = create_model(model_name, config.IN_CHANNELS, config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloaders
    print("\nLoading datasets...")
    train_dataset = XBDTilesDataset(config.TRAIN_CSV, augment=True, seed=config.SEED)
    val_dataset = XBDTilesDataset(config.VAL_CSV, augment=False, seed=config.SEED)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                           shuffle=False, num_workers=config.NUM_WORKERS)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Compute class weights
    class_weights = None
    if config.COMPUTE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(train_loader, config.NUM_CLASSES, config.DEVICE)

    # Create loss function
    criterion = CombinedLoss(
        num_classes=config.NUM_CLASSES,
        ce_weight=config.CE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        class_weights=class_weights,
        exclude_bg=config.DICE_EXCLUDE_BG
    )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    # Training loop
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LR}")
    print(f"Early stopping patience: {config.PATIENCE}\n")

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []
    }

    start_time = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        print(f"Epoch {epoch:2d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint_path = os.path.join(model_dir, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': vars(config)
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

    total_time = time.time() - start_time

    # Save final model
    final_checkpoint_path = os.path.join(model_dir, f"{model_name}_final.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'config': vars(config)
    }, final_checkpoint_path)

    # Save training history
    history_path = os.path.join(model_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save training summary
    summary = {
        'model': model_name,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'epochs_trained': epoch,
        'total_training_time': total_time,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'final_val_loss': val_loss,
        'final_val_acc': val_acc
    }

    summary_path = os.path.join(model_dir, f"{model_name}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_dir}")
    print(f"{'='*80}\n")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train segmentation models for damage assessment")
    parser.add_argument('--model', type=str, required=True,
                       choices=['fcn8s', 'unet', 'deeplabv3plus'],
                       help='Model architecture to train')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: 1e-3)')

    args = parser.parse_args()

    # Create config
    config = Config()

    # Override config with command line args if provided
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.lr is not None:
        config.LR = args.lr

    # Train model
    train_model(args.model, config)


if __name__ == "__main__":
    main()
