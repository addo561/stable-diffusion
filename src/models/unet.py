import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SpatialTransformer
from typing import List

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

class ResBlock(nn.Module):
    """Processes spatial image features and injects time context.

    Attributes:
        in_ch : input channel dim
        out_ch : output channel dim
    """
    def __init__(self,in_ch,d_t_embed,dropout=0.,out_ch=None):
        super().__init__()
        self.embed_dim = d_t_embed
        self.in_channels = in_ch
        self.out_channels = out_ch  if out_ch is not None  else  in_ch
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
        h =  self.in_layers(x)
        # project time context
        proj  = self.emb_layers(time_context)[:,:,None,None] #(b,feats,1,1)
        h =  h + proj
        h =  self.out_layers(h)
        return self.skip_connection(x) + h
        

class Upsample(nn.Module):
    def __init__(self,channels,with_conv : bool = True):
        super().__init__()
        self.with_conv  = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                                channels,
                                channels,
                                kernel_size=3,
                                stride = 1,
                                padding = 1
                                ) 
    def forward(self,x):
        x  = F.interpolate(x,scale_factor=2,mode='nearest')
        return self.conv(x) if self.with_conv else x   
   
class Downsample(nn.Module):
    def __init__(self,channels,with_conv :  bool = True):
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
        return  self.conv(x) 
     
class TimestepEmbedSequential(nn.Sequential):
    def  forward(self,x,embed,cond=None):
        for layer in self:
            if isinstance(layer,ResBlock):
                x = layer(x,embed)
            elif isinstance(layer,SpatialTransformer):
                x = layer(x,cond)
            else:
                x  = layer(x)
        return x   
     


class UNetConditional2D(nn.Module):
    """Encoder-decoder network  to  predict the noise  from  latents
    """
    def __init__(self,
                channels : int, # same as first val in block_out_channels
                in_channels : int,
                out_channels : int,
                block_out_channels : List[int], #Channel depth for each of the four resolution stages.
                layers_per_block : int, # number of resnetblocks in each up or down/up block
                levels : int ,# Number of  levels 
                attn_levels : List[int],
                cross_attention_dim : int = 768,
                n_heads  : int = 8
                 ):
        super().__init__()
        
        
        # Time embedding
        self.time_embedding = TimestepEmbedding(channels,channels*4)
        # Encoder
        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimestepEmbedSequential(
            nn.Conv2d(in_channels,channels,3,padding=1) # Projecting input tensor
        ))
        input_block_channels = [channels]
        for i in range(levels):
            for  _ in  range(layers_per_block):
                # for each level(down block types ) add resnetblocks (2) ,some attention blocks and downsample at  the end
                #  taking last  element block_out_channels as d_t_embed
                layers = [ResBlock(in_ch=channels,d_t_embed=block_out_channels[-1],out_ch=block_out_channels[i])]
                channels = block_out_channels[i]
                
                if i in attn_levels:
                    layers.append(SpatialTransformer(channels, n_heads, head_dim=80, context_dim=cross_attention_dim)) 
                    
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels) # all input block channels  , use later for decoder(skip connections)
                
            if i != levels-1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(channels)))
                input_block_channels.append(channels)
         
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels,block_out_channels[-1]),
            SpatialTransformer(channels,n_heads=n_heads,head_dim=80,context_dim=cross_attention_dim),
            ResBlock(channels,block_out_channels[-1]),
        )   
        # (decoder) 
        self.output_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            for j in range(layers_per_block+1):
                #skip connection at the resnet block
                layers = [ResBlock(in_ch=channels + input_block_channels.pop(),d_t_embed=block_out_channels[-1],out_ch=block_out_channels[i])]
                channels = block_out_channels[i]
                if i in attn_levels:
                    layers.append(SpatialTransformer(channels,n_heads,head_dim=80,context_dim=cross_attention_dim))
                if i != 0 and j == layers_per_block:
                    layers.append(Upsample(channels))    
                self.output_blocks.append(TimestepEmbedSequential(*layers))   
                
        self.out = nn.Sequential(
            nn.GroupNorm(32,channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )    
        
        
    def forward(self, x : torch.Tensor, t_steps : torch.Tensor, cond : torch.Tensor):
        input_block = []
        t_embedding  = self.time_embedding(t_steps) # (b,embed_dim)
        for m in  self.input_blocks:
            x =  m(x, t_embedding,cond) # input  modules and  what  each takes
            input_block.append(x)
        x = self.middle_block(x, t_embedding, cond)   
        for m in self.output_blocks:
            x = torch.cat([x,input_block.pop()],dim=1) # skip connections in  the  unet for decoder side(u)
            x  = m(x,t_embedding,cond)
        return  self.out(x)    
            
        

