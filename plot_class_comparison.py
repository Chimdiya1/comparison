import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_class_iou_comparison():
    RESULTS_DIR = "/content/assessment/results"
    MODELS = ['fcn8s', 'unet', 'deeplabv3plus']
    CLASS_NAMES = ["Background", "No Damage", "Minor", "Major", "Destroyed"]
    
    # Data storage
    all_data = {}

    # Load data for each model
    for model in MODELS:
        path = os.path.join(RESULTS_DIR, model, f"{model}_best_metrics.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                all_data[model] = data['class_iou']
        else:
            print(f"Warning: No best_metrics.json found for {model}")

    if not all_data:
        return

    # Plotting setup
    x = np.arange(len(CLASS_NAMES))  # Label locations
    width = 0.25  # Width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 7))

    # Define colors for the models
    model_colors = {'fcn8s': '#95a5a6', 'unet': '#3498db', 'deeplabv3plus': '#e74c3c'}

    for model_name, ious in all_data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, ious, width, label=model_name.upper(), color=model_colors.get(model_name))
        # Add labels on top of bars
        ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=9)
        multiplier += 1

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Intersection over Union (IoU)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison by Damage Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x + width, CLASS_NAMES)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylim(0, 1.1)  # IoU is 0 to 1
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Highlight the "Destroyed" class
    ax.get_xticklabels()[4].set_color('red')
    ax.get_xticklabels()[4].set_weight('bold')

    save_path = os.path.join(RESULTS_DIR, "class_iou_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Class-wise comparison saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_class_iou_comparison()