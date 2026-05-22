import torch
import torch.nn  as nn
  
class Clip_VAE(nn.Module):
    def __init__(self,
                model_name:  str,
                device,
                tokenizer = None,
                text_encoder = None,
                Vae = None 
                 ):
        """Module for clip tokenizer,text_encoder and vae decoder/encoder

        Attributes:
            text (str): specify component type(CLIP / VAE)
            device (cuda/cpu): device
            tokenizer : CLIPTokenizer, for input_ids or tokens
            text_encoder : CLIPTextModel, to  encode ids or  tokens to embeddings
            Vae : AutoencoderKL
        """
        super().__init__()
        self.model_name =  model_name
        self.device = device
        self.tokenizer  = tokenizer
        self.text_encoder = text_encoder
        self.Vae = Vae
        
        if self.model_name =='clip':
            self.get_tokens = self.tokenizer.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='tokenizer')
            self.get_embeddings = self.text_encoder.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='text_encoder').to(self.device)
            
        if  self.model_name == 'Vae_encode' or self.model_name == 'Vae_decode':
            self.latent_scaling_factor =  0.18215
            self.autoencoder = self.Vae.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='vae').to(device)
    @torch.inference_mode()        
    def forward(self,input : str | torch.Tensor)-> torch.Tensor:
        if self.model_name=='clip':
            input_ids = self.get_tokens([input],
                                       padding ='max_length',
                                       max_length = 77, #hardcoded as 77
                                       truncation=True, 
                                       return_tensors="pt",
                                       )['input_ids'] 
            input_ids_tensor =  input_ids.to(self.device)  
            text_embeddings =   self.get_embeddings(input_ids_tensor)['last_hidden_state']
            return  text_embeddings
        input = input.to(self.device)
        if  self.model_name == 'Vae_encode':
            encoded = self.latent_scaling_factor * self.autoencoder.encode(input).sample()
            return encoded
        if  self.model_name == 'Vae_decode':
            decoded  = self.autoencoder.decode(input/self.latent_scaling_factor).sample()
            return (decoded/2 + 0.5).clamp(0,1) # [-1,1] to [0,1]