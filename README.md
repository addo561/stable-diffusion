# 🎨 Stable Diffusion Pipeline

A modular implementation of Stable Diffusion v1.4 inference with custom DDIM sampling, 
classifier-free guidance, and extensible inpainting capabilities.

## 🚀 What This Project Does

Generate high-quality images from text prompts using a custom-built sampling pipeline, 
with plans for targeted inpainting.

## 🛠️ What I Built From Scratch

- **DDIM Sampler**: Complete noise scheduling and denoising loop
- **Classifier-Free Guidance**: Custom implementation of conditional/unconditional steering
- **VAE Interface**: Latent encoding/decoding with proper scaling
- **CLIP Text Pipeline**: Tokenization and embedding extraction
- **Inpainting Logic**: Mask-based latent blending (in progress)

## 🤝 What's Integrated

- **UNet Backbone**: `UNet2DConditionModel` from 🤗 Diffusers (pre-trained weights)

## 💡 Why This Approach

Initial work focused on full weight injection into a custom UNet architecture. 
While 652/656 layers loaded successfully, architectural differences (GEGLU/GELU, 
upsampling ordering) prevented coherent outputs. This led to the pragmatic decision 
to use the proven Diffusers UNet as a reliable foundation while keeping all other 
components custom — ensuring quality without compromising learning.

## 📦 Key Features

- ✅ Text-to-image generation
- ✅ Configurable steps and guidance scale
- ✅ Custom DDIM sampling loop
- ✅ Modular design (swap any component)
- 🔜 Inpainting with custom masks

## 🏗️ Architecture

### Inference Loop Diagram

<img width="680" height="582" alt="sd_inference_loop" src="https://github.com/user-attachments/assets/06010fd7-035d-467f-a56d-f835ab1801d9" />

### Custom UNet vs. Diffusers UNet – Results Comparison(prompt =  'an astronuat  riding a horse)

| My Custom UNet (weight injection attempt) | Diffusers UNet (final pipeline) |
|:-----------------------------------------:|:-------------------------------:|
| <img width="512" height="512" alt="image_generated_final-2" src="https://github.com/user-attachments/assets/706cfb48-bda0-44b5-b8ba-47b1c613ef4a" />|<img width="512" height="512" alt="image_generated_final-10" src="https://github.com/user-attachments/assets/38f4f630-d950-4c8d-a9d8-bd1849e78466" />|
| *Garbled / incoherent output* | *Coherent, prompt‑following output* |


## 🔧 Inference

```bash
python inference.py -c "your prompt" -s 50 -g 7.5
