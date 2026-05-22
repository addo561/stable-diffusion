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
alpha_bars =  scheduler.alphas_cumprod.to(device) # get alpha_bars (scheduler) from diffusers from ddim
logging.info('SCHEDULER READY')

# Get various modules
logging.info('LOADING OTHER MODELS')

model = UNetConditional2D(
        channels = 320,
        in_channels = 4,
        out_channels = 4,
        block_out_channels = [320,640,1280,1280],
        layers_per_block = 2,
        levels = 4,
        attn_levels = [0,1,2,3]
)
model.load_state_dict(torch.load('/content/stable_diffusion/model/model.pth',map_location=device))#  in colab runtime
model = model.to(device)
logging.info('UNET  MODEL READY')

# Get  embedding  for  the  text(prompts)
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
def  main():
    model.eval()
    timesteps = torch.linspace(T - 1, 0, steps, dtype=int)
    xt = torch.randn(1,4,64,64).to(device) #  only 1 sample, gaussain latent
    for t_idx in tqdm.tqdm(timesteps,desc='loading',colour='blue'):
        t  = torch.full((xt.shape[0],),t_idx,device=device).long()
        
        # Pass prompts to get conditioned  and unconditioned  predictions,  but encode them first 
        cond  = encode_text(prompt).to(device)
        cond_neg = encode_text(negative_prompt)
        conditioned_pred = model(xt,t,cond)
        unconditioned_pred = model(xt,t,cond_neg)
        
        # Guide image
        noise_pred = unconditioned_pred + guidance  * (conditioned_pred - unconditioned_pred)
        t_prev  = torch.clamp(t-10,min=0)
        alpha_bars_t = alpha_bars[t].reshape(-1,1,1,1)
        alpha_bars_prev  =  alpha_bars[t_prev].reshape(-1,1,1,1)
        predict_x_0 =( xt  - (((1  -  alpha_bars_t)**0.5) * noise_pred)) /  (alpha_bars_t**0.5)
        predict_x_0 = predict_x_0.clamp(-1,1)
        pointing_xt  =   ((1 - alpha_bars_prev)**0.5) * noise_pred
        xt = (alpha_bars_prev **0.5) * predict_x_0 +  pointing_xt

    # After getting prediction,decode with vae  and save
    decoded = decoder_vae(xt)
    decoded =  decoded.cpu().permute(0,2,3,1).float().numpy()  # range  [0,1]
    decoded =  (decoded[0] * 255)
    decoded = decoded.astype('uint8') 
    path = Path('outputs')
    path.mkdir(exist_ok=True)
    img_path = path / 'image_generated_1.png'
    plt.imsave(img_path,decoded)
    logging.info(f'Saved image to {img_path}')
    
if __name__ =='__main__':
    main()