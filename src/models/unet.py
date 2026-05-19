import torch
import torch.nn as nn

## UNEt
class  TimeEmbedding(nn.Module):
    """Encodes scalar diffusion steps into continuous vectors.
    """
    def  __init__(self,base_dim,f_dim):
        super().__init__()
        self.base_dim = base_dim # or d_model
        self.f_dim=  f_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(self.base_dim,self.f_dim),
            nn.SiLU(),
            nn.Linear(self.f_dim,self.f_dim)
        )
    def forward(self,timesteps: torch.Tensor)-> torch.Tensor:
        time_vector = torch.arange(0,len(timesteps),device=timesteps.device).float()#(len(steps),)
        dim_vector = torch.arange(0,self.base_dim,2).float()#(base_dim/2,)
        frequency_scales = 10000 ** (dim_vector/self.base_dim)
        f_s =  1/frequency_scales
        #final = time_vector[None,:].T @ f_s[None,:]  # (steps,1) x (1,base_dim/2)
        final = torch.outer(time_vector,f_s)#(steps,base_dim/2)
        embedding = torch.zeros(len(timesteps),self.base_dim) #(steps,base_dim)
        embedding[:,0::2] = torch.sin(final)
        embedding[:,1::2]  = torch.cos(final)
        context =  self.mlp(embedding)  #(dim or d_model,final dim)
        return context


class ResnetBlock(nn.Module):
    """Processes spatial image features and injects time context.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,):
        super().__init__()    