import torch
import torch.nn as nn
import random

class ResBlock(nn.Module):
    """ResNet residual block with reflection padding."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """
    CycleGAN Generator: ResNet-based.
    Uses InstanceNorm and ReflectionPad.
    """
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=6):
        super().__init__()
        # Encoder
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf,    ngf*2,  3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ngf*2,  ngf*4,  3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(inplace=True),
        ]
        
        # Transformation
        for _ in range(n_blocks):
            layers.append(ResBlock(ngf*4))

        # Decoder
        layers += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(ngf*2, ngf,   3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """70x70 PatchGAN Discriminator for CycleGAN (Unconditional)."""
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        def conv_block(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *conv_block(in_ch, ndf,    norm=False),
            *conv_block(ndf,   ndf*2),
            *conv_block(ndf*2, ndf*4),
            *conv_block(ndf*4, ndf*8, stride=1),
            nn.Conv2d(ndf*8, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class ImageBuffer:
    """Replay buffer for stable discriminator training."""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    result.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    result.append(element)
        return torch.stack(result)
