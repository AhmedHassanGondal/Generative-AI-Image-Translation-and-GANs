# 🎓 Viva Preparation Guide: Generative AI Pipeline

This guide prepares you for common questions asked during a technical defense of your GAN project.

---

## 1. Top 10 Viva Questions & Professional Answers

### Q1: Why did you use WGAN-GP instead of a standard DCGAN for some tasks?
**Answer:** Standard GANs (DCGAN) use Binary Cross Entropy (BCE) loss, which often leads to **vanishing gradients** and **mode collapse**. WGAN-GP (Wasserstein GAN with Gradient Penalty) uses the **Earth-Mover (Wasserstein-1) distance**, providing a continuous and meaningful loss signal. The **Gradient Penalty** ensures the critic satisfies the 1-Lipschitz constraint, leading to much more stable training.

### Q2: Explain the "U-Net" architecture in Pix2Pix. Why not use a standard Encoder-Decoder?
**Answer:** In image-to-image translation, a standard Encoder-Decoder loses high-frequency spatial information (like sharp edges) during downsampling. **U-Net** introduces **skip connections** between the encoder and decoder layers. This allows the model to pass low-level structural details directly to the output layers, resulting in sharper translations.

### Q3: What is "Cycle Consistency Loss" in CycleGAN?
**Answer:** In unpaired data (like Horse ↔ Zebra), we don't have direct ground truth. Cycle consistency ensures that if we translate an image from Domain A to Domain B and then back to Domain A, we should get the original image back ($A \rightarrow G_{AB}(A) \rightarrow G_{BA}(G_{AB}(A)) \approx A$). This constrains the model so it doesn't just generate any random realistic image but one that preserves the input's structure.

### Q4: What is a "PatchGAN" Discriminator?
**Answer:** Unlike a standard discriminator that outputs a single scalar (Real/Fake) for the whole image, **PatchGAN** outputs a grid (e.g., $30 \times 30$). Each pixel in this grid represents whether a $70 \times 70$ patch of the input image is real or fake. This forces the generator to focus on local textures and sharp details.

### Q5: How do you handle "Mode Collapse"?
**Answer:** We handle it through:
1. **WGAN-GP loss** (more stable distance metric).
2. **Instance Normalization** (better for generative tasks than Batch Normalization).
3. **Identity Loss** in CycleGAN (preserves target domain attributes).
4. **Replay Buffers** in CycleGAN (prevents the discriminator from oscillating by showing it older fake images).

### Q6: What is the role of "Identity Loss"?
**Answer:** It ensures that if the generator receives an image that is already in the target domain, it leaves it unchanged ($G_{AB}(B) \approx B$). This helps in preserving the original colors and global structure of the input image.

### Q7: Why did you use InstanceNorm instead of BatchNorm for CycleGAN?
**Answer:** In generative tasks, Batch Normalization can introduce unwanted artifacts because it depends on the statistics of the entire batch. **Instance Normalization** normalizes each image independently, which is better for style transfer and image-to-image translation where batch sizes are often very small (e.g., $batch\_size=1$).

### Q8: What metrics did you use to evaluate your models?
**Answer:** We used **SSIM (Structural Similarity Index)** and **PSNR (Peak Signal-to-Noise Ratio)**.
- **SSIM**: Measures perceived image quality based on luminance, contrast, and structure (closer to 1 is better).
- **PSNR**: Measures pixel-wise reconstruction quality (higher is better).

### Q9: How did you optimize the pipeline for Kaggle GPUs?
**Answer:** I implemented **Mixed Precision Training (torch.amp)**, which uses 16-bit floats for computations while keeping weights in 32-bit. This reduces memory usage and speeds up training by nearly 2x on T4/A100 GPUs without sacrificing accuracy.

### Q10: What is the "Replay Buffer"?
**Answer:** It’s a technique where we store a history of generated images. During discriminator training, we show it a mix of current fakes and older fakes from the buffer. This prevents the discriminator from "forgetting" what older fake images looked like and stabilizes the adversarial game.

---

## 💡 Pro-Tip for your Presentation:
When explaining the code, point to the **Modular Structure**. Tell the examiners: *"I moved away from flat notebooks to a modular structure to ensure the project is scalable and production-ready, following industry best practices like separating configuration from model logic."*
