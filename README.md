# 🎨 Stable Diffusion Pipeline

A hands-on implementation of Stable Diffusion v1.4 inference with custom DDIM sampling,
classifier-free guidance, and inpainting — built piece by piece to understand what's actually happening under the hood.

---

## 🖼️ Inpainting Result

> Full walkthrough in [`in-painting.ipynb`](./in-painting.ipynb)

<!-- Replace with your inpainting output image -->
<img width="512" height="512" alt="__results___9_12" src="https://github.com/user-attachments/assets/b4df05f5-3ee7-4b81-9470-2fb43227cd13" />


*Mask-based latent blending: the original image is preserved outside the mask, and new content is diffused inside it.*

---

## 🚀 What This Project Does

Generate images from text prompts using a custom-built sampling pipeline, with working inpainting on top.

---

## 🛠️ What I Built From Scratch

- **DDIM Sampler** — complete noise scheduling and denoising loop
- **Classifier-Free Guidance** — custom conditional/unconditional steering
- **VAE Interface** — latent encoding/decoding with proper scaling
- **CLIP Text Pipeline** — tokenization and embedding extraction
- **Inpainting Logic** — mask-based latent blending (`in-painting.ipynb`)

## 🤝 What's Integrated

- **UNet Backbone** — `UNet2DConditionModel` from 🤗 Diffusers (pre-trained weights)

---

## 💡 Why This Approach

Initial work focused on injecting weights into a fully custom UNet architecture.
684/686 layers loaded successfully, but architectural mismatches (GEGLU vs GELU activations,
upsampling order) prevented coherent outputs. Rather than paper over the issue,
the pragmatic call was to use the proven Diffusers UNet as a stable backbone while keeping
every other component custom — quality without sacrificing what was learned.

> See [`stable-diffusion.ipynb`](./stable-diffusion.ipynb) for that experiment.

---

## 🏗️ Architecture

### Inference Loop

<img width="680" height="582" alt="sd_inference_loop" src="https://github.com/user-attachments/assets/06010fd7-035d-467f-a56d-f835ab1801d9" />

---

### Custom UNet vs. Diffusers UNet

Prompt: *"an astronaut riding a horse"* — 35 steps each

| My Custom UNet (weight injection attempt) | Diffusers UNet (final pipeline) |
|:-----------------------------------------:|:-------------------------------:|
| <img width="512" height="512" alt="custom unet output" src="https://github.com/user-attachments/assets/706cfb48-bda0-44b5-b8ba-47b1c613ef4a" /> | <img width="512" height="512" alt="diffusers unet output" src="https://github.com/user-attachments/assets/38f4f630-d950-4c8d-a9d8-bd1849e78466" /> |
| *Garbled / incoherent output* | *Coherent, prompt-following output* |

---

## 📦 Features

- ✅ Text-to-image generation
- ✅ Configurable steps and guidance scale
- ✅ Custom DDIM sampling loop
- ✅ Inpainting with custom masks (`in-painting.ipynb`)

---

## 🔧 Usage

```bash
python inference.py -c "your prompt" -s 50 -g 7.5
```<img width="512" height="512" alt="__results___9_12" src="https://github.com/user-attachments/assets/c5168e58-7eee-426e-99fd-4c7c80b13542" />
