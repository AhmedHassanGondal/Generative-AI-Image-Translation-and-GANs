import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..utils.metrics import calculate_metrics
from ..utils.visualization import save_image_grid
import os

class Pix2PixTrainer(BaseTrainer):
    """Trainer for Pix2Pix (Conditional GAN)."""
    def __init__(self, generator, discriminator, config, device):
        super().__init__(config, device)
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        # Optimizers
        self.opt_G = optim.Adam(self.generator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        # Losses
        self.criterion_GAN = nn.MSELoss() # LSGAN for stability
        self.criterion_L1 = nn.L1Loss()
        self.lambda_L1 = config.get('lambda_L1', 100.0)

    def train_epoch(self, dataloader):
        self.generator.train()
        self.discriminator.train()
        
        total_loss_G = 0
        total_loss_D = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for i, (sketch, color) in enumerate(pbar):
            sketch, color = sketch.to(self.device), color.to(self.device)
            
            # ── Update Discriminator ──
            self.opt_D.zero_grad()
            with autocast(device_type='cuda'):
                # Real
                pred_real = self.discriminator(color, sketch)
                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
                
                # Fake
                fake_color = self.generator(sketch)
                pred_fake = self.discriminator(fake_color.detach(), sketch)
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                
            self.scaler.scale(loss_D).backward()
            self.scaler.step(self.opt_D)
            
            # ── Update Generator ──
            self.opt_G.zero_grad()
            with autocast(device_type='cuda'):
                # Adversarial loss
                pred_fake = self.discriminator(fake_color, sketch)
                loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                
                # L1 Pixel loss
                loss_G_L1 = self.criterion_L1(fake_color, color) * self.lambda_L1
                
                loss_G = loss_G_GAN + loss_G_L1
                
            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.opt_G)
            
            self.scaler.update()
            
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            
            pbar.set_postfix({"G": loss_G.item(), "D": loss_D.item()})
            
            # Visualise every 500 batches
            if i % 500 == 0:
                save_image_grid(torch.cat([sketch[:2], fake_color[:2], color[:2]]), 
                                os.path.join(self.vis_dir, f"train_ep{self.current_epoch}_batch{i}.png"),
                                title="Sketch | Generated | Real")
                
        return {"G": total_loss_G/len(dataloader), "D": total_loss_D/len(dataloader)}

    @torch.no_grad()
    def validate(self, dataloader):
        self.generator.eval()
        ssims, psnrs = [], []
        
        for sketch, color in tqdm(dataloader, desc="Validating"):
            sketch, color = sketch.to(self.device), color.to(self.device)
            fake_color = self.generator(sketch)
            
            s, p = calculate_metrics(color, fake_color)
            ssims.append(s); psnrs.append(p)
            
        avg_ssim = sum(ssims)/len(ssims)
        avg_psnr = sum(psnrs)/len(psnrs)
        
        # Save a sample from validation
        save_image_grid(torch.cat([sketch[:2], fake_color[:2], color[:2]]), 
                        os.path.join(self.vis_dir, f"val_epoch_{self.current_epoch}.png"),
                        title="Sketch | Generated | Real")
        
        return {"SSIM": avg_ssim, "PSNR": avg_psnr}

    def save_checkpoint(self, path):
        torch.save({
            'epoch': self.current_epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'scaler': self.scaler.state_dict()
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt['generator'])
        self.discriminator.load_state_dict(ckpt['discriminator'])
        self.opt_G.load_state_dict(ckpt['opt_G'])
        self.opt_D.load_state_dict(ckpt['opt_D'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.current_epoch = ckpt['epoch']
