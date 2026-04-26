# Generative AI Pipeline: GANs, Pix2Pix & CycleGAN

A modular, professional implementation of state-of-the-art Generative Adversarial Networks (GANs) using PyTorch. This repository provides unified training pipelines, evaluation metrics, and visualization for four key architectures.

## 🚀 Key Features

- **DCGAN**: Baseline deep convolutional GAN for image generation.
- **WGAN-GP**: Stable training using Wasserstein loss and Gradient Penalty to mitigate mode collapse.
- **Pix2Pix**: Paired image-to-image translation using U-Net generator and PatchGAN discriminator.
- **CycleGAN**: Unpaired domain adaptation using cycle consistency and identity losses.
- **Unified Workflow**: Standardized trainers, mixed-precision training (AMP), and integrated SSIM/PSNR metrics.
- **Kaggle Optimized**: Designed for high-performance training on Kaggle T4x2 GPUs.

## 📂 Project Structure

```text
GAN_Experiments-main/
├── configs/            # Experiment-specific configurations
├── core/
│   ├── data/           # Unified datasets (Paired, Unpaired, Single Domain)
│   ├── models/         # Architecture definitions (U-Net, ResNet, PatchGAN)
│   ├── trainers/       # Modular trainers (Pix2Pix, CycleGAN, DCGAN/WGAN)
│   └── utils/          # Metrics (SSIM/PSNR) and Visualization helpers
├── notebooks/          # Original research and experiment logs
├── main.py             # Unified entry point
└── requirements.txt    # Project dependencies
```

## 🛠️ Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training

Train any model using the `main.py` entry point. Specify the mode and data paths.

#### **Pix2Pix (Paired)**
```bash
python main.py --mode pix2pix --sketch_dir path/to/sketches --color_dir path/to/colors
```

#### **CycleGAN (Unpaired)**
```bash
python main.py --mode cyclegan --dir_a path/to/domain_A --dir_b path/to/domain_B
```

#### **WGAN-GP (Generation)**
```bash
python main.py --mode wgan-gp --data_root path/to/images
```

## 📊 Evaluation & Visualization

- **Metrics**: SSIM (Structural Similarity Index) and PSNR (Peak Signal-to-Noise Ratio) are automatically calculated during validation for translation tasks.
- **Visuals**: Real-time visualization grids are saved in `outputs/visualizations/` during training to monitor progress.

## 🏗️ Model Architectures

- **U-Net Generator**: Used in Pix2Pix for high-resolution skip connections.
- **ResNet Generator**: 6/9 ResNet blocks with reflection padding for CycleGAN stability.
- **PatchGAN Discriminator**: 70x70 receptive field to penalize local structure.
- **Wasserstein Critic**: Critic model with InstanceNorm for stable generation in WGAN-GP.

## ⚖️ License
MIT
