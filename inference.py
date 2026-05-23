##  DDIM(sampling)
# use diffusers scheduler  for alpha_prod_t  and alpha_prod_t_prev 
#   DDIM sampler
#   1. Look at xt
#   2. Predict the noise
#   3. Estimate the clean image
#   4. Re-noise it for the next timestep

import tqdm
import torch
from diffusers import  DDIMScheduler
from transformers import CLIPTokenizer,CLIPTextModel
from diffusers import AutoencoderKL
import argparse
from src.vae_clip import Clip_VAE
from src.unet import UNetConditional2D
from config import config
from transformers import CLIPTokenizer
import matplotlib.pyplot  as plt
from pathlib  import Path 
import  logging
device = 'cuda' if torch.cuda.is_available()  else 'cpu'


#  logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M", #edit  format asctime
    level='INFO',
    force=True
    )


parser = argparse.ArgumentParser(
    description = 'Generate image'
)
parser.add_argument('-c','--cond', type = str ,)
parser.add_argument('-c_n','--c_neg',type = str,default = 'blurry image, low quality')
parser.add_argument('-s','--steps', type =  int)
parser.add_argument('-g','--guidance', type  = float, default = 7.5, help='Classifier-free guidance scale')
args = parser.parse_args()

logging.info('LOADING SCHEDULER')
model_pretrained = config['compvis']['pretrained_model_name_or_path']
scheduler = DDIMScheduler.from_pretrained(model_pretrained, subfolder = 'scheduler')
logging.info('SCHEDULER READY')

#  various modules
logging.info('LOADING OTHER MODELS')

model = UNetConditional2D(
        channels = 320,
        in_channels = 4,
        out_channels = 4,
        block_out_channels = [320,640,1280,1280],
        layers_per_block = 2,
        levels = 4,
        attn_levels = [0,1,2]
)
model.load_state_dict(torch.load('/content/stable_diffusion/model/model_f.pth',map_location=device))#  in colab runtime
logging.info(model.state_dict()['input_blocks.1.0.in_layers.2.weight'].mean().item())
model = model.to(device)
logging.info('UNET  MODEL READY')

#   embedding  for  the  text(prompts)
encode_text = Clip_VAE(
                    model_name = 'clip',
                    device = device,
                    tokenizer = CLIPTokenizer,
                    text_encoder = CLIPTextModel
                    )

#  Decoder  vae
decoder_vae = Clip_VAE(
                    model_name = 'Vae_decode',
                    device = device,
                    Vae=AutoencoderKL
                    )
logging.info('CLIP AND VAE READY')

# Steps,prompt and negative prompt,timesteps
steps = args.steps
prompt = args.cond 
negative_prompt = args.c_neg
T = config['timesteps']['T']
guidance = args.guidance

### -------- WHOLE INFERENCE  LOOP ------------ ### 
@torch.inference_mode()
def main():
    model.eval()
    #  number of inference steps in scheduler
    scheduler.set_timesteps(steps, device=device)
    timesteps = scheduler.timesteps  # automatically descending, [999, 979, ..., 0]
    xt = torch.randn(1, 4, 64, 64).to(device)
    # encode 
    cond = encode_text(prompt).to(device)
    cond_neg = encode_text(negative_prompt).to(device)   

    for t_idx in tqdm.tqdm(timesteps, desc='loading', colour='blue'):
        t = torch.full((xt.shape[0],), t_idx, device=device).long()

        conditioned_pred = model(xt, t, cond)
        unconditioned_pred = model(xt, t, cond_neg)
        # apply cfg
        noise_pred = unconditioned_pred + guidance * (conditioned_pred - unconditioned_pred)
        xt = scheduler.step(noise_pred, t, xt).prev_sample

    # get decoded  image
    decoded = decoder_vae(xt)               
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
    decoded = (decoded[0] * 255).astype('uint8')

    path = Path('outputs')
    path.mkdir(exist_ok=True)
    img_path = path / 'image_generated_final.png'
    plt.imsave(img_path, decoded)
    logging.info(f'Saved image to {img_path}')

if __name__ == '__main__':
    main()
