import os
import torch
from abc import ABC, abstractmethod
from torch.amp import GradScaler, autocast
from tqdm import tqdm

class BaseTrainer(ABC):
    """Abstract Base Class for GAN Trainers."""
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.scaler = GradScaler('cuda')
        self.current_epoch = 0
        
        # Directories
        self.ckpt_dir = os.path.join(config.get('out_dir', 'outputs'), 'checkpoints')
        self.vis_dir = os.path.join(config.get('out_dir', 'outputs'), 'visualizations')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    @abstractmethod
    def train_epoch(self, dataloader):
        """Implement logic for one training epoch."""
        pass

    @abstractmethod
    def validate(self, dataloader):
        """Implement logic for validation/evaluation."""
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        """Save model weights and optimizer states."""
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """Load model weights and optimizer states."""
        pass

    def train(self, tr_loader, va_loader=None, num_epochs=None):
        """Master training loop."""
        epochs = num_epochs or self.config.get('epochs', 10)
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch + 1
            print(f"\nEpoch {self.current_epoch}/{epochs}")
            
            train_losses = self.train_epoch(tr_loader)
            print(f"Train Losses: {train_losses}")
            
            if va_loader:
                val_metrics = self.validate(va_loader)
                print(f"Validation Metrics: {val_metrics}")
            
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(os.path.join(self.ckpt_dir, f"epoch_{epoch+1}.pt"))
