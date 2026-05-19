from transformers import CLIPTokenizer,CLIPTextModel
from diffusers import AutoencoderKL
import torch
import torch.nn  as nn
  
class Clip_VAE(nn.Module):
    def __init__(self,
                 model_name:  str,
                 device
                 ):
        """Module for clip tokenizer,text_encoder and vae decoder/encoder

        Args:
            text (str): specify component type(CLIP / VAE)
            device (cuda/cpu): device
        """
        super().__init__()
        self.model_name =  model_name
        self.device = device
        if self.model_name =='clip':
            self.tokenizer = CLIPTokenizer.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='tokenizer')
            self.text_encoder = CLIPTextModel.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='text_encoder').to(self.device)
            
        if  self.model_name == 'Vae_encode':
            pass
        if self.model_name == 'Vae_decode':
            pass    
    def forward(self,input)-> torch.Tensor:
        if self.model_name=='clip':
            input_ids = self.tokenizer([input],
                                       paddings='max_length',
                                       max_length = self.tokenizer.model_max_length,
                                       truncation=True, 
                                       return_tensors="pt",
                                       )['input_ids'] 
            input_ids_tensor =  input_ids.to(self.device)  
            with torch.no_grad():
                text_embeddings =   self.text_encoder(input_ids_tensor)['last_hidden_state']
            return  text_embeddings.shape
            