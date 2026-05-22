# Stable Diffusion — Custom UNet with CompVis v1.4 Weights

A text-to-image pipeline built in PyTorch from scratch. The UNet denoising
network is a custom implementation with weights injected from the CompVis
Stable Diffusion v1.4 checkpoint. All other components are loaded directly
from CompVis v1.4.

---

## Pipeline Components

| Component | Source | Notes |
|---|---|---|
| **UNet** | Custom (this repo) | Built from scratch, weights injected from CompVis v1.4 safetensors |
| **VAE** | CompVis v1.4 | Encodes and decodes latents — scale factor `0.18215` |
| **CLIP Text Encoder** | CompVis v1.4 | ViT-L/14, produces 768-dim context embeddings |
| **Tokenizer** | CompVis v1.4 | CLIP BPE, max 77 tokens |
| **Scheduler** | DDIM | 20–50 steps sufficient for good quality |

---

## UNet Architecture

<img width="680" height="620" alt="unet_encoder_decoder_loop" src="https://github.com/user-attachments/assets/bf313e8d-0715-4b9a-8dfe-953d2b1a185d" />

---

## Weight Injection

Weights are loaded from `clean_compvis_v14_unet.safetensors` — a pre-cleaned
flat dict with no Lightning wrapper and no `model.diffusion_model.` prefix.

Key translation rules applied at load time:

```
time_embed.            →  time_embedding.mlp.
.op.weight / .op.bias  →  .conv.weight / .conv.bias
.ff.net.0.proj.        →  .ff.0.
.ff.net.2.             →  .ff.2.
```

The CompVis feed-forward uses **GEGLU** activation — `ff.net.0.proj` weight
shape is `[5120, dim]`. Since this UNet uses plain **GELU**, only the first
half `[:2560]` is loaded and the gate half is discarded.

---

## Inference

```bash
python -m inference -c "your prompt here" -s 35
```

<img width="680" height="582" alt="sd_inference_loop" src="https://github.com/user-attachments/assets/06010fd7-035d-467f-a56d-f835ab1801d9" />

---

## Dependencies

```
diffusers>=0.36.0
einops>=0.8.2
lightning>=2.6.0
matplotlib>=3.9.4
numpy<2
path>=17.1.1
torch>=2.2.2
torchinfo>=1.8.0
torchvision>=0.17.2
tqdm>=4.67.3
transformers>=4.57.6
```

---

## References

- [CompVis Stable Diffusion v1.4](https://github.com/CompVis/stable-diffusion)
- [DDIM — Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [CLIP — Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
