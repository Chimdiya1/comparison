import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
RESULTS_DIR = "/content/assessment/results"
MODELS = ['fcn8s', 'unet', 'deeplabv3plus']
CLASS_NAMES = ["Background", "No Damage", "Minor", "Major", "Destroyed"]

def load_history(model_name):
    path = os.path.join(RESULTS_DIR, model_name, f"{model_name}_history.json")
    if not os.path.exists(path):
        print(f"Warning: No history found for {model_name}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def plot_comparison():
    plt.style.use('ggplot')
    
    # 1. NEW: Mean IoU vs Epochs
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        hist = load_history(model)
        if hist and 'val_miou' in hist:
            plt.plot(hist['val_miou'], label=f"{model.upper()} (Max: {max(hist['val_miou']):.4f})", linewidth=2)
    
    plt.title("Segmentation Performance: Mean IoU (mIoU) per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("mIoU Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_miou.png"))
    plt.show()

    # 2. Validation Loss Comparison (Existing)
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        hist = load_history(model)
        if hist:
            plt.plot(hist['val_loss'], label=f"{model.upper()} (Min: {min(hist['val_loss']):.4f})", linewidth=2)
    
    plt.title("Model Convergence: Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Combined Loss (CE + Dice)")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_val_loss.png"))
    plt.show()

    # 3. Efficiency Plot (Time vs mIoU)
    plt.figure(figsize=(10, 6))
    for model in MODELS:
        hist = load_history(model)
        if hist and 'val_miou' in hist:
            # Calculate cumulative training time in minutes
            times = [sum(hist['epoch_times'][:i+1])/60 for i in range(len(hist['epoch_times']))]
            plt.plot(times, hist['val_miou'], label=f"{model.upper()}", marker='o', markersize=4)
            
    plt.title("Architecture Efficiency: mIoU vs. Total Training Time")
    plt.xlabel("Total Training Time (Minutes)")
    plt.ylabel("mIoU")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_efficiency_miou.png"))
    plt.show()

if __name__ == "__main__":
    plot_comparison()