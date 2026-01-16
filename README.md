# Building Damage Assessment - Model Comparison

This directory contains the implementation and experiments for comparing three semantic segmentation architectures (FCN-8s, UNet, and DeepLabV3+) for building damage assessment from satellite imagery.

## Directory Structure

```
assessment/
├── models/
│   ├── fcn.py              # FCN-8s implementation
│   ├── deeplabv3plus.py    # DeepLabV3+ implementation
│   └── __init__.py
├── results/                # Training outputs and metrics (created during training)
│   ├── fcn8s/
│   ├── unet/
│   └── deeplabv3plus/
├── figures/                # Visualizations for paper
├── train_comparison.py     # Unified training script
├── MODEL_SUMMARY.md        # Detailed model documentation
└── README.md              # This file
```

## Quick Start

### 1. Training

Train each model with identical hyperparameters for fair comparison:

```bash
# Activate virtual environment
source venv/bin/activate

# Train FCN-8s
python assessment/train_comparison.py --model fcn8s

# Train UNet
python assessment/train_comparison.py --model unet

# Train DeepLabV3+
python assessment/train_comparison.py --model deeplabv3plus
```

### 2. Training Configuration

All models use **identical hyperparameters**:
- **Batch size:** 8
- **Learning rate:** 1e-3
- **Optimizer:** Adam
- **Epochs:** 20 (with early stopping, patience=5)
- **Loss:** Combined Cross-Entropy + Dice Loss
- **Data augmentation:** Flip, rotation, brightness/contrast
- **Random seed:** 42 (for reproducibility)

### 3. Optional Arguments

You can override default settings:

```bash
# Custom batch size
python assessment/train_comparison.py --model unet --batch-size 4

# Custom epochs
python assessment/train_comparison.py --model fcn8s --epochs 15

# Custom learning rate
python assessment/train_comparison.py --model deeplabv3plus --lr 5e-4
```

## Output Files

After training each model, the following files are saved in `results/<model_name>/`:

- **`<model>_best.pth`** - Best model checkpoint (lowest validation loss)
- **`<model>_final.pth`** - Final model checkpoint
- **`<model>_history.json`** - Training curves (loss, accuracy per epoch)
- **`<model>_summary.json`** - Training summary (parameters, time, metrics)

Example `<model>_summary.json`:
```json
{
  "model": "unet",
  "total_parameters": 7764037,
  "trainable_parameters": 7764037,
  "epochs_trained": 15,
  "total_training_time": 1234.56,
  "best_val_loss": 0.1234,
  "best_val_acc": 0.8765,
  "final_val_loss": 0.1298,
  "final_val_acc": 0.8723
}
```

## Models

### FCN-8s
- **Backbone:** VGG16
- **Parameters:** ~134.3M
- **Key Feature:** Element-wise skip connections from pool3, pool4, and final layer

### UNet
- **Backbone:** Custom encoder-decoder
- **Parameters:** ~7.8M
- **Key Feature:** Symmetric architecture with concatenation skip connections

### DeepLabV3+
- **Backbone:** ResNet50
- **Parameters:** ~40.4M
- **Key Feature:** ASPP module with atrous convolutions for multi-scale context

## Dataset

- **Source:** xView2 Hurricane Harvey
- **Training samples:** ~1,278 tiles (256×256)
- **Validation samples:** ~319 tiles
- **Classes:** 5 (background, no-damage, minor, major, destroyed)
- **Input:** 6 channels (pre + post RGB concatenated)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- NumPy
- Pillow
- CUDA (recommended for GPU training)

## Reproducibility

All experiments use:
- **Fixed seed (42)** for random number generators
- **Deterministic CUDA operations** (when available)
- **Identical data splits** from CSV files
- **Same augmentation pipeline** for all models



