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
from src.models.unet import UNetConditional2D
import argparse
from config import config
device = 'cuda' if torch.cuda.is_available()  else 'cpu'

parser = argparse.ArgumentParser(
    description = 'Generate image'
)
parser.add_argument('-cond', type = str ,)
parser.add_argument('-c_neg',
                    type = str,
                    default = 'blurry image, low quality'
                    )
parser.add_argument('-steps', type =  int)
args = parser.parse_args()

model_pretrained = config['compvis']['pretrained_model_name_or_path']
scheduler = DDIMScheduler.from_pretrained(model_pretrained, subfolder = 'scheduler',device = device)
alpha_bars =  scheduler.alphas_cumprod # get alpha_bars (scheduler) from diffusers from ddim

# instantiate model
model = UNetConditional2D(
    channels = 320,
    in_channels = 4,
    out_channels = 4,
    block_out_channels = [320,640,1280,1280],
    layers_per_block = 2,
    levels = 4,
    attn_levels = [0,1,2,3]
).to(device)

#steps,prompt and negative prompt,timesteps
steps = args.steps
prompt = args.cond 
negative_prompt = args.c_neg
T = config['timesteps']['T']

### -------- WHOLE INFERENCE  LOOP ------------ ### 

with torch.inference_mode:
    model.eval()
    for t_idx in tqdm.tqdm(reversed(range(0,T,steps)),desc='loading',colour='blue'):
        t  = torch.full((xt.shape[0],),t_idx,device=device).long()
        t_prev  = torch.clamp(t-10,min=0)
        alpha_bars = alpha_bars[t].reshape(-1,1,1,1)
        alpha_bars_prev  =  alpha_bars[t_prev].reshape(-1,1,1,1)
        noise_pred = model(xt,t).sample
        predict_x_0 =( xt  - (((1  -  alpha_bars)**0.5) * noise_pred)) /  (alpha_bars**0.5)
        predict_x_0 = predict_x_0.clamp(-1,1)
        pointing_xt  =   ((1 - alpha_bars_prev)**0.5) * noise_pred
        xt = (alpha_bars_prev **0.5) * predict_x_0 +  pointing_xt


if __name__ =='__main__':
    