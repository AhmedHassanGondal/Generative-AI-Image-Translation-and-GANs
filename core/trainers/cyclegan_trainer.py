import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm
from .base_trainer import BaseTrainer
from ..utils.metrics import calculate_metrics
from ..utils.visualization import save_image_grid
from ..models.cyclegan import ImageBuffer
import os
import itertools

class CycleGANTrainer(BaseTrainer):
    """Trainer for CycleGAN (Unpaired Domain Adaptation)."""
    def __init__(self, G_AB, G_BA, D_A, D_B, config, device):
        super().__init__(config, device)
        self.G_AB = G_AB.to(device) # Domain A -> B
        self.G_BA = G_BA.to(device) # Domain B -> A
        self.D_A = D_A.to(device)   # Discriminator for A
        self.D_B = D_B.to(device)   # Discriminator for B
        
        # Optimizers
        self.opt_G = optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), 
                                lr=config['lr'], betas=(0.5, 0.999))
        self.opt_D_A = optim.Adam(self.D_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.opt_D_B = optim.Adam(self.D_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        
        # Losses
        self.criterion_GAN = nn.MSELoss() 
        self.criterion_cycle = nn.L1Loss()
        self.criterion_id = nn.L1Loss()
        
        self.lambda_cycle = config.get('lambda_cycle', 10.0)
        self.lambda_id = config.get('lambda_id', 5.0)
        
        # Replay Buffers
        self.buffer_A = ImageBuffer(config.get('buffer_size', 50))
        self.buffer_B = ImageBuffer(config.get('buffer_size', 50))

    def train_epoch(self, dataloader):
        self.G_AB.train(); self.G_BA.train()
        self.D_A.train(); self.D_B.train()
        
        total_loss_G = 0
        total_loss_D = 0
        
        pbar = tqdm(dataloader, desc="Training CycleGAN")
        for i, (real_A, real_B) in enumerate(pbar):
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            
            # ── Update Generators ──
            self.opt_G.zero_grad()
            with autocast(device_type='cuda'):
                # Identity loss (preserve colors/style)
                id_A = self.G_BA(real_A)
                loss_id_A = self.criterion_id(id_A, real_A) * self.lambda_id
                
                id_B = self.G_AB(real_B)
                loss_id_B = self.criterion_id(id_B, real_B) * self.lambda_id
                
                # GAN loss
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
                
                fake_A = self.G_BA(real_B)
                loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
                
                # Cycle loss
                rec_A = self.G_BA(fake_B)
                loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_cycle
                
                rec_B = self.G_AB(fake_A)
                loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_cycle
                
                # Total Generator Loss
                loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
                
            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.opt_G)
            
            # ── Update Discriminator A ──
            self.opt_D_A.zero_grad()
            with autocast(device_type='cuda'):
                loss_D_A_real = self.criterion_GAN(self.D_A(real_A), torch.ones_like(self.D_A(real_A)))
                
                fake_A_buf = self.buffer_A.push_and_pop(fake_A.detach())
                loss_D_A_fake = self.criterion_GAN(self.D_A(fake_A_buf), torch.zeros_like(self.D_A(fake_A_buf)))
                
                loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
                
            self.scaler.scale(loss_D_A).backward()
            self.scaler.step(self.opt_D_A)
            
            # ── Update Discriminator B ──
            self.opt_D_B.zero_grad()
            with autocast(device_type='cuda'):
                loss_D_B_real = self.criterion_GAN(self.D_B(real_B), torch.ones_like(self.D_B(real_B)))
                
                fake_B_buf = self.buffer_B.push_and_pop(fake_B.detach())
                loss_D_B_fake = self.criterion_GAN(self.D_B(fake_B_buf), torch.zeros_like(self.D_B(fake_B_buf)))
                
                loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
                
            self.scaler.scale(loss_D_B).backward()
            self.scaler.step(self.opt_D_B)
            
            self.scaler.update()
            
            total_loss_G += loss_G.item()
            total_loss_D += (loss_D_A.item() + loss_D_B.item())
            
            pbar.set_postfix({"G": loss_G.item(), "D": (loss_D_A.item() + loss_D_B.item())})
            
            if i % 500 == 0:
                save_image_grid(torch.cat([real_A[:1], fake_B[:1], rec_A[:1], real_B[:1], fake_A[:1], rec_B[:1]]), 
                                os.path.join(self.vis_dir, f"train_ep{self.current_epoch}_batch{i}.png"),
                                title="A | A->B | A->B->A || B | B->A | B->A->B", n_rows=2)
                
        return {"G": total_loss_G/len(dataloader), "D": total_loss_D/len(dataloader)}

    @torch.no_grad()
    def validate(self, dataloader):
        self.G_AB.eval(); self.G_BA.eval()
        # Evaluation for CycleGAN often uses cycle consistency (SSIM between real and reconstructed)
        ssims_A, ssims_B = [], []
        
        for real_A, real_B in tqdm(dataloader, desc="Validating"):
            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            rec_A = self.G_BA(self.G_AB(real_A))
            rec_B = self.G_AB(self.G_BA(real_B))
            
            sA, _ = calculate_metrics(real_A, rec_A)
            sB, _ = calculate_metrics(real_B, rec_B)
            ssims_A.append(sA); ssims_B.append(sB)
            
        avg_ssim = (sum(ssims_A)/len(ssims_A) + sum(ssims_B)/len(ssims_B)) / 2
        
        save_image_grid(torch.cat([real_A[:1], self.G_AB(real_A)[:1], rec_A[:1], real_B[:1], self.G_BA(real_B)[:1], rec_B[:1]]), 
                        os.path.join(self.vis_dir, f"val_epoch_{self.current_epoch}.png"),
                        title="A | A->B | rec_A || B | B->A | rec_B", n_rows=2)
        
        return {"Avg_SSIM_Recon": avg_ssim}

    def save_checkpoint(self, path):
        torch.save({
            'epoch': self.current_epoch,
            'G_AB': self.G_AB.state_dict(),
            'G_BA': self.G_BA.state_dict(),
            'D_A': self.D_A.state_dict(),
            'D_B': self.D_B.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D_A': self.opt_D_A.state_dict(),
            'opt_D_B': self.opt_D_B.state_dict(),
            'scaler': self.scaler.state_dict()
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.G_AB.load_state_dict(ckpt['G_AB'])
        self.G_BA.load_state_dict(ckpt['G_BA'])
        self.D_A.load_state_dict(ckpt['D_A'])
        self.D_B.load_state_dict(ckpt['D_B'])
        self.opt_G.load_state_dict(ckpt['opt_G'])
        self.opt_D_A.load_state_dict(ckpt['opt_D_A'])
        self.opt_D_B.load_state_dict(ckpt['opt_D_B'])
        self.scaler.load_state_dict(ckpt['scaler'])
        self.current_epoch = ckpt['epoch']
