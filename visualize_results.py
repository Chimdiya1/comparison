import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Assuming your Config and Model classes are in the same directory or accessible
from train_comparison import Config, XBDTilesDataset, create_model

def colorize_mask(mask):
    """Converts a (H, W) label mask to an (H, W, 3) RGB image"""
    cmap = np.array([
        [0, 0, 0],       # 0: Background - Black
        [0, 255, 0],     # 1: No Damage - Green
        [255, 255, 0],   # 2: Minor - Yellow
        [255, 165, 0],   # 3: Major - Orange
        [255, 0, 0]      # 4: Destroyed - Red
    ], dtype=np.uint8)
    return cmap[mask]

def run_visual_comparison(num_samples=5):
    config = Config()
    device = config.DEVICE
    
    # 1. Load Data
    val_dataset = XBDTilesDataset(config.VAL_CSV, augment=False)
    
    # 2. Load Models
    models = {}
    model_names = ['fcn8s', 'unet', 'deeplabv3plus']
    
    for name in model_names:
        print(f"Loading best {name} model...")
        model = create_model(name, config.IN_CHANNELS, config.NUM_CLASSES)
        checkpoint_path = os.path.join(config.OUT_DIR, name, f"{name}_best.pth")
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device).eval()
            models[name] = model
        else:
            print(f"Warning: Checkpoint not found for {name}")

    # 3. Pick samples and Plot
    # We look for samples that actually have buildings (non-zero pixels)
    indices = []
    for i in range(len(val_dataset)):
        _, mask = val_dataset[i]
        if torch.sum(mask > 0) > 1000: # Ensure at least some buildings are present
            indices.append(i)
            if len(indices) >= num_samples: break

    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    cols = ["Post-Image", "Ground Truth", "FCN-8s", "UNet", "DeepLabV3+"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=16, fontweight='bold')

    for i, idx in enumerate(indices):
        x_t, y_t = val_dataset[idx]
        
        # Display Post-disaster Image (Channels 3, 4, 5 in your 6-ch input)
        post_img = x_t[3:].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(post_img)
        
        # Display Ground Truth
        axes[i, 1].imshow(colorize_mask(y_t.numpy()))
        
        # Display Model Predictions
        input_batch = x_t.unsqueeze(0).to(device)
        for m_idx, name in enumerate(model_names):
            if name in models:
                with torch.no_grad():
                    logits = models[name](input_batch)
                    pred = torch.argmax(logits, dim=1).cpu().squeeze(0).numpy()
                axes[i, m_idx + 2].imshow(colorize_mask(pred))
            else:
                axes[i, m_idx + 2].text(0.5, 0.5, "N/A", ha='center')

        # Clean up axes
        for j in range(5):
            axes[i, j].axis('off')

    save_path = os.path.join(config.OUT_DIR, "model_visual_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Comparison saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_visual_comparison()