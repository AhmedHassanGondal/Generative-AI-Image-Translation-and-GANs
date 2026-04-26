import gradio as gr
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from core.models.pix2pix import Pix2PixGenerator
from core.models.cyclegan import CycleGANGenerator

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(mode, checkpoint_path):
    if mode == 'pix2pix':
        model = Pix2PixGenerator(3, 3, 64)
    elif mode == 'cyclegan':
        model = CycleGANGenerator(3, 3, 64, 6)
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle both full trainer checkpoints and state_dicts
    state_dict = ckpt['generator'] if 'generator' in ckpt else (ckpt['G_AB'] if 'G_AB' in ckpt else ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def predict(image, mode, checkpoint):
    # Load model
    model = load_model(mode, checkpoint)
    
    # Preprocess
    img_size = 256 if mode == 'pix2pix' else 128
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    output = (output.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1)
    return (output * 255).astype(np.uint8)

# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Dropdown(["pix2pix", "cyclegan"], label="Model Mode"),
        gr.Textbox(label="Checkpoint Path", placeholder="e.g. outputs/pix2pix/checkpoints/best.pt")
    ],
    outputs=gr.Image(type="numpy", label="Generated Image"),
    title="Generative AI Pipeline: Image-to-Image Translation",
    description="Upload a sketch or photo to see the GAN in action. Supports Pix2Pix and CycleGAN."
)

if __name__ == "__main__":
    interface.launch()
