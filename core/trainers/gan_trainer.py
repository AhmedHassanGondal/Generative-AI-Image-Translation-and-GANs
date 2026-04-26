import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..utils.visualization import save_image_grid
import os

class GANTrainer(BaseTrainer):
    """Trainer for DCGAN and WGAN-GP."""
    def __init__(self, generator, discriminator, config, device, mode='dcgan'):
        super().__init__(config, device)
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.mode = mode.lower()
        self.nz = config.get('nz', 100)
        
        # Optimizers
        lr = config.get('lr', 0.0002)
        betas = (0.5, 0.999) if self.mode == 'dcgan' else (0.0, 0.9)
        self.opt_G = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        
        # Losses
        self.criterion = nn.BCELoss() if self.mode == 'dcgan' else None
        self.lambda_gp = config.get('lambda_gp', 10.0)

    def gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty for WGAN-GP."""
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(d_interpolates.shape, device=self.device, requires_grad=False)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_epoch(self, dataloader):
        self.generator.train()
        self.discriminator.train()
        
        total_loss_G = 0
        total_loss_D = 0
        
        pbar = tqdm(dataloader, desc=f"Training {self.mode.upper()}")
        for i, real_img in enumerate(pbar):
            batch_size = real_img.size(0)
            real_img = real_img.to(self.device)
            
            # ── Update Discriminator ──
            self.opt_D.zero_grad()
            with autocast(device_type='cuda'):
                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
                fake_img = self.generator(noise)
                
                if self.mode == 'dcgan':
                    # Real
                    output_real = self.discriminator(real_img).view(-1)
                    loss_D_real = self.criterion(output_real, torch.ones_like(output_real))
                    # Fake
                    output_fake = self.discriminator(fake_img.detach()).view(-1)
                    loss_D_fake = self.criterion(output_fake, torch.zeros_like(output_fake))
                    loss_D = loss_D_real + loss_D_fake
                else: # wgan-gp
                    loss_D_real = -torch.mean(self.discriminator(real_img))
                    loss_D_fake = torch.mean(self.discriminator(fake_img.detach()))
                    gp = self.gradient_penalty(real_img, fake_img.detach())
                    loss_D = loss_D_real + loss_D_fake + self.lambda_gp * gp
                
            self.scaler.scale(loss_D).backward()
            self.scaler.step(self.opt_D)
            
            # ── Update Generator ──
            self.opt_G.zero_grad()
            with autocast(device_type='cuda'):
                if self.mode == 'dcgan':
                    output = self.discriminator(fake_img).view(-1)
                    loss_G = self.criterion(output, torch.ones_like(output))
                else: # wgan-gp
                    loss_G = -torch.mean(self.discriminator(fake_img))
                    
            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.opt_G)
            self.scaler.update()
            
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            
            pbar.set_postfix({"G": loss_G.item(), "D": loss_D.item()})
            
            if i % 1000 == 0:
                save_image_grid(fake_img[:16], 
                                os.path.join(self.vis_dir, f"fake_ep{self.current_epoch}_batch{i}.png"),
                                title=f"Generated {self.mode.upper()}", n_rows=4)
                
        return {"G": total_loss_G/len(dataloader), "D": total_loss_D/len(dataloader)}

    @torch.no_grad()
    def validate(self, dataloader):
        # For pure GANs, validation is mostly visual
        noise = torch.randn(16, self.nz, 1, 1, self.device)
        fake_img = self.generator(noise)
        path = os.path.join(self.vis_dir, f"val_epoch_{self.current_epoch}.png")
        save_image_grid(fake_img, path, title=f"Validation {self.mode.upper()}", n_rows=4)
        return {"visual_saved": path}

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
