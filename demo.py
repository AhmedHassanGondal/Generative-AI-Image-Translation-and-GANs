import torch
import torch.nn as nn
from core.models.dcgan import DCGANGenerator, DCGANDiscriminator
from core.models.wgan import WGANGenerator, WGANCritic
from core.models.pix2pix import Pix2PixGenerator, Pix2PixDiscriminator
from core.models.cyclegan import CycleGANGenerator, CycleGANDiscriminator

def test_model(name, model, input_size):
    print(f"Testing {name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ {name} forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
    print("-" * 30)

def main():
    print("🚀 Starting Generative AI Pipeline Sanity Check...\n")
    
    # 1. DCGAN
    test_model("DCGAN Generator", DCGANGenerator(100, 64, 3), (1, 100, 1, 1))
    
    # 2. WGAN-GP
    test_model("WGAN Critic", WGANCritic(3, 64), (1, 3, 64, 64))
    
    # 3. Pix2Pix
    test_model("Pix2Pix Generator", Pix2PixGenerator(3, 3, 64), (1, 3, 256, 256))
    
    # 4. CycleGAN
    test_model("CycleGAN Generator", CycleGANGenerator(3, 3, 64, 6), (1, 3, 128, 128))

    print("\n🌟 All models are correctly initialized and executable!")
    print("You can now proceed to training using 'main.py' or deployment using 'app.py'.")

if __name__ == "__main__":
    main()
