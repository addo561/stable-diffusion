import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SpatialTransformer

## UNEt
class  TimestepEmbedding(nn.Module):
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
        time_vector = timesteps.flatten().float()#(len(steps),)
        b =  time_vector.shape[0]
        dim_vector = torch.arange(0,self.base_dim,2,device=timesteps.device).float()#(base_dim/2,)
        frequency_scales = 10000 ** (dim_vector/self.base_dim)
        f_s =  1/frequency_scales
        #final = time_vector[:,None] @ f_s[None,:]  # (steps,1) x (1,base_dim/2)
        final = torch.outer(time_vector,f_s)#(steps,base_dim/2)
        embedding = torch.zeros(b,self.base_dim,device=timesteps.device) #(steps,base_dim)
        embedding[:,0::2] = torch.sin(final)
        embedding[:,1::2]  = torch.cos(final)
        context =  self.mlp(embedding)  #(steps,final dim)
        return context

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResBlock(nn.Module): # name have to match compVis class
    """Processes spatial image features and injects time context.

    Attributes:
        in_ch : input channel dim
        out_ch : output channel dim
        d_t_embed : embed_channels
        dropout
    """
    def __init__(self,in_ch,d_t_embed,out_ch,dropout=0.):
        super().__init__()
        self.embed_dim = d_t_embed
        self.in_channels = in_ch
        self.out_channels = out_ch  
        self.in_layers = nn.Sequential(
                Normalize(in_channels=self.in_channels),
                nn.SiLU(),
                nn.Conv2d(
                        self.in_channels,
                        self.out_channels,
                        kernel_size=3,
                        padding=1,
                    )
        )
        
        # Project time embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim,self.out_channels)
            
            )
        self.out_layers = nn.Sequential(
                Normalize(in_channels=self.out_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Conv2d(
                        self.out_channels,
                        self.out_channels,
                        kernel_size=3,
                        padding=1,
                    )
        )
        
        if self.in_channels == self.out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                                    self.in_channels,
                                    self.out_channels, 
                                    1
                                    )    
    def forward(self,x,time_context):
        """
            (x)Image feats: (Batch, In_Channels, H, W)
            Time context: (Batch, 1280)
        """
        h = x
        h =  self.in_layers(h)
        # project time context
        proj  = self.emb_layers(time_context)[:,:,None,None] #(b,feats,1,1)
        h =  h + proj
        h =  self.out_layers(h)
        return self.skip_connection(x) + h
        


class Upsample(nn.Module):
    def __init__(self,channels,with_conv:bool):
        super().__init__()
        self.with_conv  = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                                channels,
                                channels,
                                kernel_size=3,
                                stride = 1,
                                ) 
    def forward(self,x):
        x  = F.interpolate(x,scale_factor=2,mode='nearest')
        return self.conv(x) if self.with_conv else x   
   
class Downsample(nn.Module):
    def __init__(self,channels,with_conv:bool):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                                channels,
                                channels,
                                kernel_size=3,
                                stride=2,
                                padding=1)
    def forward(self,x):
        if self.with_conv:
            pad = (1,0,1,0)
            x = F.pad(x,pad=pad,mode='constant',value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x,kernel_size=3,stride=2)     
        return  x  
     
     
class UNetConditional2D(nn.Module):
    def __init__(self,
                 in_channels=4,
                 out_channels=4,
                 model_channels=320):
        super().__init__()
        
        # Time Encoder Track
        self.time_embed = TimestepEmbedding(base_dim=320, out_dim=1280)
        
        #  The Encoder Track 
        self.input_blocks = nn.ModuleList([
            
        ])
        
        # bottleneck
        self.middle_block = nn.Sequential(
            ResBlock(1280,1280,1280,dropout=0.1),
            SpatialTransformer(channels=1280, n_heads=8, head_dim=160),
            ResBlock(1280,1280,1280,dropout=0.1)
        )
        
        #  The Decoder Track
        self.output_blocks = nn.ModuleList([ ... 12 Sequential Slots ... ])
        
        #  The Output 
        self.out = nn.Sequential(
            nn.GroupNorm(32, 320),
            nn.SiLU(),
            nn.Conv2d(320, out_channels, kernel_size=3, padding=1)
        )
    def forward(self,):
        pass    
