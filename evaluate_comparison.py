"""
Evaluation script for comparing FCN-8s, UNet, and DeepLabV3+ models.

This script:
1. Loads all three trained models
2. Computes comprehensive metrics on the validation/test set
3. Generates comparison tables (for paper)
4. Measures inference time
5. Saves results to JSON and CSV for easy analysis

Usage:
    python assessment/evaluate_comparison.py
"""

import os
import sys
import json
import time
import csv
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from train_comparison import Config, XBDTilesDataset, create_model


# ----------------------------
# Metrics Classes
# ----------------------------
class SegmentationMetrics:
    """Comprehensive metrics for semantic segmentation evaluation."""

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        """Reset all accumulated statistics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_pixels = 0
        self.correct_pixels = 0

    def update(self, predictions, targets):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            predictions: (N, H, W) tensor of predicted class labels
            targets: (N, H, W) tensor of ground truth labels
        """
        preds = predictions.cpu().numpy().flatten()
        targs = targets.cpu().numpy().flatten()

        # Update confusion matrix
        mask = (targs >= 0) & (targs < self.num_classes)
        self.confusion_matrix += np.bincount(
            self.num_classes * targs[mask].astype(int) + preds[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

        # Update pixel accuracy
        self.total_pixels += mask.sum()
        self.correct_pixels += (preds[mask] == targs[mask]).sum()

    def compute(self):
        """
        Compute all metrics from accumulated statistics.

        Returns:
            dict: Dictionary containing all computed metrics
        """
        results = {}

        # Pixel Accuracy
        results['pixel_accuracy'] = self.correct_pixels / (self.total_pixels + 1e-10)

        # Per-class metrics from confusion matrix
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp

        # IoU per class
        iou = tp / (tp + fp + fn + 1e-10)
        results['iou_per_class'] = iou.tolist()
        results['mean_iou'] = np.nanmean(iou)

        # F1 per class (Dice coefficient)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        results['precision_per_class'] = precision.tolist()
        results['recall_per_class'] = recall.tolist()
        results['f1_per_class'] = f1.tolist()
        results['mean_f1'] = np.nanmean(f1)
        results['macro_f1'] = np.mean(f1)

        # Weighted F1 (by support)
        support = self.confusion_matrix.sum(axis=1)
        weighted_f1 = np.average(f1, weights=support)
        results['weighted_f1'] = weighted_f1

        # Mean precision and recall
        results['mean_precision'] = np.nanmean(precision)
        results['mean_recall'] = np.nanmean(recall)

        # Confusion matrix (normalized)
        cm_normalized = self.confusion_matrix.astype(float) / (
            self.confusion_matrix.sum(axis=1, keepdims=True) + 1e-10
        )
        results['confusion_matrix'] = self.confusion_matrix.tolist()
        results['confusion_matrix_normalized'] = cm_normalized.tolist()

        return results


def evaluate_model(model, dataloader, device, num_classes, class_names):
    """
    Evaluate a single model on the dataset.

    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run inference on
        num_classes: Number of classes
        class_names: List of class names

    Returns:
        dict: Dictionary containing all metrics and timing info
    """
    model.eval()
    metrics = SegmentationMetrics(num_classes, class_names)

    inference_times = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # Measure inference time
            start_time = time.time()
            logits = model(x)
            torch.cuda.synchronize() if device == 'cuda' else None
            inference_times.append(time.time() - start_time)

            # Get predictions
            preds = torch.argmax(logits, dim=1)

            # Update metrics
            metrics.update(preds, y)

    # Compute final metrics
    results = metrics.compute()

    # Add timing info
    results['total_inference_time'] = sum(inference_times)
    results['avg_inference_time_per_batch'] = np.mean(inference_times)
    results['avg_inference_time_per_image'] = np.mean(inference_times) / dataloader.batch_size
    results['num_batches'] = len(inference_times)
    results['num_samples'] = len(dataloader.dataset)

    return results, metrics.confusion_matrix


def count_parameters(model):
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def plot_confusion_matrices(confusion_matrices, model_names, class_names, save_path):
    """
    Plot confusion matrices for all models side by side.

    Args:
        confusion_matrices: dict of model_name -> confusion_matrix
        model_names: list of model names
        class_names: list of class names
        save_path: path to save the figure
    """
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        cm = confusion_matrices[name]
        # Normalize
        cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{name.upper()} Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def plot_metrics_comparison(all_results, model_names, save_dir):
    """
    Create bar charts comparing metrics across models.

    Args:
        all_results: dict of model_name -> results
        model_names: list of model names
        save_dir: directory to save figures
    """
    # Overall metrics comparison
    metrics_to_compare = ['pixel_accuracy', 'mean_iou', 'mean_f1', 'macro_f1']
    metric_labels = ['Pixel Accuracy', 'Mean IoU', 'Mean F1', 'Macro F1']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_to_compare))
    width = 0.25

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, name in enumerate(model_names):
        values = [all_results[name][m] for m in metrics_to_compare]
        bars = ax.bar(x + i * width, values, width, label=name.upper(), color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'overall_metrics_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overall metrics comparison saved to {save_path}")

    # Per-class IoU comparison
    class_names = ["Background", "No Damage", "Minor", "Major", "Destroyed"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.25

    for i, name in enumerate(model_names):
        values = all_results[name]['iou_per_class']
        bars = ax.bar(x + i * width, values, width, label=name.upper(), color=colors[i])

    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'per_class_iou_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class IoU comparison saved to {save_path}")

    # Per-class F1 comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(model_names):
        values = all_results[name]['f1_per_class']
        bars = ax.bar(x + i * width, values, width, label=name.upper(), color=colors[i])

    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'per_class_f1_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class F1 comparison saved to {save_path}")


def generate_latex_tables(all_results, model_names, model_params, save_dir):
    """
    Generate LaTeX tables for the paper.

    Args:
        all_results: dict of model_name -> results
        model_names: list of model names
        model_params: dict of model_name -> (total_params, trainable_params)
        save_dir: directory to save tables
    """
    class_names = ["Background", "No Damage", "Minor", "Major", "Destroyed"]

    # Table 1: Overall Performance
    table1_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Overall Performance Comparison}",
        "\\label{tab:overall}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Model & Params (M) & Pixel Acc & Mean IoU & Mean F1 & Macro F1 & Inf. Time (ms) \\\\",
        "\\hline"
    ]

    for name in model_names:
        r = all_results[name]
        params = model_params[name][0] / 1e6  # Convert to millions
        inf_time = r['avg_inference_time_per_image'] * 1000  # Convert to ms

        line = f"{name.upper()} & {params:.1f} & {r['pixel_accuracy']:.4f} & {r['mean_iou']:.4f} & {r['mean_f1']:.4f} & {r['macro_f1']:.4f} & {inf_time:.1f} \\\\"
        table1_lines.append(line)

    table1_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])

    with open(os.path.join(save_dir, 'table1_overall.tex'), 'w') as f:
        f.write('\n'.join(table1_lines))

    # Table 2: Per-Class IoU
    table2_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Per-Class IoU Scores}",
        "\\label{tab:iou}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Model & Background & No Damage & Minor & Major & Destroyed \\\\",
        "\\hline"
    ]

    for name in model_names:
        iou = all_results[name]['iou_per_class']
        line = f"{name.upper()} & {iou[0]:.4f} & {iou[1]:.4f} & {iou[2]:.4f} & {iou[3]:.4f} & {iou[4]:.4f} \\\\"
        table2_lines.append(line)

    table2_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])

    with open(os.path.join(save_dir, 'table2_iou.tex'), 'w') as f:
        f.write('\n'.join(table2_lines))

    # Table 3: Per-Class F1
    table3_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Per-Class F1 Scores}",
        "\\label{tab:f1}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Model & Background & No Damage & Minor & Major & Destroyed \\\\",
        "\\hline"
    ]

    for name in model_names:
        f1 = all_results[name]['f1_per_class']
        line = f"{name.upper()} & {f1[0]:.4f} & {f1[1]:.4f} & {f1[2]:.4f} & {f1[3]:.4f} & {f1[4]:.4f} \\\\"
        table3_lines.append(line)

    table3_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])

    with open(os.path.join(save_dir, 'table3_f1.tex'), 'w') as f:
        f.write('\n'.join(table3_lines))

    print(f"LaTeX tables saved to {save_dir}")


def generate_csv_summary(all_results, model_names, model_params, save_path):
    """
    Generate CSV summary for easy analysis.

    Args:
        all_results: dict of model_name -> results
        model_names: list of model names
        model_params: dict of model_name -> (total_params, trainable_params)
        save_path: path to save CSV
    """
    class_names = ["Background", "No Damage", "Minor", "Major", "Destroyed"]

    rows = []
    for name in model_names:
        r = all_results[name]
        row = {
            'Model': name.upper(),
            'Total Parameters': model_params[name][0],
            'Parameters (M)': model_params[name][0] / 1e6,
            'Pixel Accuracy': r['pixel_accuracy'],
            'Mean IoU': r['mean_iou'],
            'Mean F1': r['mean_f1'],
            'Macro F1': r['macro_f1'],
            'Weighted F1': r['weighted_f1'],
            'Mean Precision': r['mean_precision'],
            'Mean Recall': r['mean_recall'],
            'Inference Time (ms/image)': r['avg_inference_time_per_image'] * 1000,
        }

        # Add per-class metrics
        for i, cls_name in enumerate(class_names):
            row[f'IoU_{cls_name}'] = r['iou_per_class'][i]
            row[f'F1_{cls_name}'] = r['f1_per_class'][i]
            row[f'Precision_{cls_name}'] = r['precision_per_class'][i]
            row[f'Recall_{cls_name}'] = r['recall_per_class'][i]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"CSV summary saved to {save_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare segmentation models")
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()

    # Configuration
    config = Config()
    device = config.DEVICE

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config.OUT_DIR, 'evaluation')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)

    class_names = ["Background", "No Damage", "Minor", "Major", "Destroyed"]
    model_names = ['fcn8s', 'unet', 'deeplabv3plus']

    print("="*80)
    print("Model Evaluation and Comparison")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print()

    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = XBDTilesDataset(config.VAL_CSV, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=config.NUM_WORKERS)
    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Evaluate each model
    all_results = {}
    model_params = {}
    confusion_matrices = {}

    for name in model_names:
        print(f"\n{'='*60}")
        print(f"Evaluating {name.upper()}")
        print(f"{'='*60}")

        # Load model
        model = create_model(name, config.IN_CHANNELS, config.NUM_CLASSES)
        checkpoint_path = os.path.join(config.OUT_DIR, name, f"{name}_best.pth")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Skipping this model...")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Count parameters
        total_params, trainable_params = count_parameters(model)
        model_params[name] = (total_params, trainable_params)
        print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

        # Evaluate
        print("Running evaluation...")
        results, cm = evaluate_model(model, val_loader, device,
                                     config.NUM_CLASSES, class_names)

        all_results[name] = results
        confusion_matrices[name] = cm

        # Print summary
        print(f"\nResults for {name.upper()}:")
        print(f"  Pixel Accuracy: {results['pixel_accuracy']:.4f}")
        print(f"  Mean IoU:       {results['mean_iou']:.4f}")
        print(f"  Mean F1:        {results['mean_f1']:.4f}")
        print(f"  Macro F1:       {results['macro_f1']:.4f}")
        print(f"  Inference time: {results['avg_inference_time_per_image']*1000:.2f} ms/image")

        print(f"\n  Per-class IoU:")
        for i, cls_name in enumerate(class_names):
            print(f"    {cls_name}: {results['iou_per_class'][i]:.4f}")

        # Clean up
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    # Check if we have results to compare
    if len(all_results) == 0:
        print("\nNo models were successfully evaluated. Exiting.")
        return

    evaluated_models = list(all_results.keys())

    print(f"\n{'='*80}")
    print("Generating Comparison Outputs")
    print(f"{'='*80}")

    # Save raw results to JSON
    json_results = {name: {k: v for k, v in r.items()
                          if not isinstance(v, np.ndarray)}
                   for name, r in all_results.items()}

    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to {os.path.join(output_dir, 'evaluation_results.json')}")

    # Generate figures
    figures_dir = os.path.join(output_dir, 'figures')

    # Confusion matrices
    plot_confusion_matrices(
        confusion_matrices, evaluated_models, class_names,
        os.path.join(figures_dir, 'confusion_matrices.png')
    )

    # Metrics comparison charts
    plot_metrics_comparison(all_results, evaluated_models, figures_dir)

    # Generate LaTeX tables
    tables_dir = os.path.join(output_dir, 'tables')
    generate_latex_tables(all_results, evaluated_models, model_params, tables_dir)

    # Generate CSV summary
    df = generate_csv_summary(
        all_results, evaluated_models, model_params,
        os.path.join(output_dir, 'evaluation_summary.csv')
    )

    # Print final comparison table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")

    print("\n" + df[['Model', 'Parameters (M)', 'Pixel Accuracy', 'Mean IoU',
                    'Mean F1', 'Inference Time (ms/image)']].to_string(index=False))

    # Determine winners
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    best_accuracy = max(evaluated_models, key=lambda x: all_results[x]['pixel_accuracy'])
    best_iou = max(evaluated_models, key=lambda x: all_results[x]['mean_iou'])
    best_f1 = max(evaluated_models, key=lambda x: all_results[x]['mean_f1'])
    fastest = min(evaluated_models, key=lambda x: all_results[x]['avg_inference_time_per_image'])
    smallest = min(evaluated_models, key=lambda x: model_params[x][0])

    print(f"\n  Best Pixel Accuracy: {best_accuracy.upper()} ({all_results[best_accuracy]['pixel_accuracy']:.4f})")
    print(f"  Best Mean IoU:       {best_iou.upper()} ({all_results[best_iou]['mean_iou']:.4f})")
    print(f"  Best Mean F1:        {best_f1.upper()} ({all_results[best_f1]['mean_f1']:.4f})")
    print(f"  Fastest Inference:   {fastest.upper()} ({all_results[fastest]['avg_inference_time_per_image']*1000:.2f} ms)")
    print(f"  Smallest Model:      {smallest.upper()} ({model_params[smallest][0]/1e6:.1f}M params)")

    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
