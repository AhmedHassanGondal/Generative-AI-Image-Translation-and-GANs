import torch
import numpy as np
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

def denormalize(tensor):
    """Converts [-1, 1] tensor to [0, 1] numpy array."""
    return (tensor.cpu().float().numpy() * 0.5 + 0.5).clip(0, 1)

def calculate_metrics(real, generated):
    """Calculates average SSIM and PSNR between real and generated batches."""
    real_np = denormalize(real).transpose(0, 2, 3, 1) # (B, H, W, C)
    gen_np = denormalize(generated).transpose(0, 2, 3, 1)
    
    ssims = []
    psnrs = []
    
    for r, g in zip(real_np, gen_np):
        ssims.append(compute_ssim(r, g, channel_axis=2, data_range=1.0))
        psnrs.append(compute_psnr(r, g, data_range=1.0))
        
    return np.mean(ssims), np.mean(psnrs)
