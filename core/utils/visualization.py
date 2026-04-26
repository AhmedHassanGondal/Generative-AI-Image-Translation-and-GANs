import matplotlib.pyplot as plt
import torch
import numpy as np

def save_image_grid(images, path, title="", n_rows=2):
    """Saves a grid of images for visualization."""
    images = images.cpu().float()
    images = (images * 0.5 + 0.5).clamp(0, 1)
    
    batch_size = images.size(0)
    n_cols = (batch_size + n_rows - 1) // n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle(title)
    
    for i in range(batch_size):
        ax = axes.flatten()[i]
        img = images[i].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_losses(losses_dict, path):
    """Plots training losses."""
    plt.figure(figsize=(10, 5))
    for label, values in losses_dict.items():
        plt.plot(values, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
