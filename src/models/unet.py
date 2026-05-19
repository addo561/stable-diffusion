import torch
import torch.nn as nn
import torch.nn.functional as F

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


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    """Processes spatial image features and injects time context.

    Attributes:
        in_ch : input channel dim
        out_ch : output channel dim
    """
    def __init__(self,in_ch,d_t_embed,out_ch,dropout):
        super().__init__()
        self.embed_dim = d_t_embed
        self.in_channels = in_ch
        self.out_channels = out_ch  
        self.inps = nn.Sequential(
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
        self.time_proj = nn.Linear(self.embed_dim,self.out_channels)
        self.outs = nn.Sequential(
                Normalize(in_channels=self.out_channels),
                nn.SiLU(),
                nn.Dropout(0.1),
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
        h =  self.inps(x)
        # project time context
        proj  = self.time_proj(time_context)[:,:,None,None] #(b,feats,1,1)
        h =  h + proj
        h =  self.outs(h)
        return self.skip_connection(x) + h
        

 
class CrossAttn(nn.Module):
    def __init__(self,context_dim,channel_dim):
        super().__init__()
        self.context_dim = context_dim
        self.q = nn.Linear(channel_dim,channel_dim,bias=False)
        self.k = nn.Linear(context_dim,channel_dim,bias=False)
        self.v = nn.Linear(context_dim,channel_dim,bias=False)
        self.out = nn.Linear(channel_dim,channel_dim)
    def forward(self,x,context_matrix):
        '''
        Image sequence(x) : [b,c,h,w]
         context  matrix : [b,seq,context_dim]
        '''
        b,c,h,w = x.size()
        x = x.permute(0,2,3,1).view(b,h*w,c)
        q  = self.q(x) #(b,seq_q,c)
        k = self.k(context_matrix)#(b,seq_k,c) 
        v = self.v(context_matrix)#(b,seq_v,c)
        scores = q @ k.transpose(-2,-1)#(b,seq_q,seq_k)
        scale = q.shape[-1] ** 0.5
        scaled = scores / scale
        prob = F.softmax(scaled,dim=-1)
        result = prob @ v #(b,seq_q,seq_k)
        return self.out(result) #(b,seq,c)

class UpSample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.out =  nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(channels,channels,3,padding=1)
    def forward(self,x):
        #x is just an image
        res  = self.out(x)
        return self.conv(res)    
   
class DownSample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,3,stride=2,padding=1)
    def forward(self,x):
        #x is just an image
        return self.conv(x)    
    