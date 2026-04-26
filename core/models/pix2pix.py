import torch
import torch.nn as nn

class DownBlock(nn.Module):
    """Encoder block: Conv(stride=2) -> BN -> LeakyReLU"""
    def __init__(self, in_c, out_c, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder block: ConvTranspose(stride=2) -> BN -> ReLU (+ optional Dropout)"""
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Pix2PixGenerator(nn.Module):
    """
    U-Net Generator for Pix2Pix.
    Architecture (256x256 input):
      Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 (bottleneck)
      Decoder: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
      Skip connections between encoder[i] and decoder[8-i]
    """
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        # Encoder
        self.e1 = DownBlock(in_ch,   ngf,     use_bn=False)  # 256->128
        self.e2 = DownBlock(ngf,     ngf*2)                  # 128->64
        self.e3 = DownBlock(ngf*2,   ngf*4)                  # 64->32
        self.e4 = DownBlock(ngf*4,   ngf*8)                  # 32->16
        self.e5 = DownBlock(ngf*8,   ngf*8)                  # 16->8
        self.e6 = DownBlock(ngf*8,   ngf*8)                  # 8->4
        self.e7 = DownBlock(ngf*8,   ngf*8)                  # 4->2
        self.e8 = DownBlock(ngf*8,   ngf*8,  use_bn=False)   # 2->1 (bottleneck)

        # Decoder
        self.d1 = UpBlock(ngf*8,    ngf*8, dropout=True)   # 1->2
        self.d2 = UpBlock(ngf*8*2,  ngf*8, dropout=True)   # 2->4
        self.d3 = UpBlock(ngf*8*2,  ngf*8, dropout=True)   # 4->8
        self.d4 = UpBlock(ngf*8*2,  ngf*8)                 # 8->16
        self.d5 = UpBlock(ngf*8*2,  ngf*4)                 # 16->32
        self.d6 = UpBlock(ngf*4*2,  ngf*2)                 # 32->64
        self.d7 = UpBlock(ngf*2*2,  ngf)                   # 64->128
        
        # Final layer
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1),    # 128->256
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # Decoder with skip connections
        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        
        return self.d8(d7)


class Pix2PixDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.
    Conditional: Takes (input_sketch, output_image) concatenated (6 ch).
    Outputs 30x30 patch logit map for 256x256 input.
    """
    def __init__(self, in_ch=6, ndf=64):
        super().__init__()
        def conv_block(in_c, out_c, stride=2, use_bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(in_ch, ndf,    use_bn=False), # 256->128
            conv_block(ndf,   ndf*2),                # 128->64
            conv_block(ndf*2, ndf*4),                # 64->32
            conv_block(ndf*4, ndf*8, stride=1),      # 32->31 (stride 1)
            nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1) # 31->30
        )

    def forward(self, x, condition):
        # Concatenate condition and image
        input_data = torch.cat([x, condition], dim=1)
        return self.model(input_data)
