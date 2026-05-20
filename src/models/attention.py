import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttn(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        context_dim = context_dim  if context_dim is not None else query_dim
        # This tells the linear layers how big the shared attention space needs to be.
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    def forward(self,x,context_matrix=None):
        '''
        Image sequence(x) : [b,pixels,c]
         context  matrix : [b,seq,context_dim]
        '''
        if context_matrix is None:
            context_matrix = x
        b,pixels,_ = x.size()
        _,context,_ = context_matrix.size()
        q  = self.to_q(x) #(b,pixels,inner_dim)
        k = self.to_k(context_matrix)#(b,seq_k,inner_dim) 
        v = self.to_v(context_matrix)#(b,seq_v,inner_dim)
        
        #Merge heads into the batch dimension # (b*h,pixels/context,dim_head)
        dim_head = q.shape[-1]//self.heads
        q  = q.view(b,pixels,self.heads,dim_head).permute(0,2,1,3).reshape(b*self.heads,pixels,dim_head)
        k  = k.view(b,context,self.heads,dim_head).permute(0,2,1,3).reshape(b*self.heads,context,dim_head)
        v  = v.view(b,context,self.heads,dim_head).permute(0,2,1,3).reshape(b*self.heads,context,dim_head)
        scores = torch.bmm(q, k.transpose(-2, -1)) * self.scale #(b*h,pixels,context)
        attn = F.softmax(scores, dim=-1) # (b*h,pixels,context)
        
        res = attn @ v #(b*h,pixels,dim_head)
        res = res.view(b,self.heads,pixels,dim_head).permute(0,2,1,3).reshape(b,pixels,-1) #(b,pixels,inner_dim)
        return self.to_out(res) #(b,pixels,query_dim)
        
    
class TransformerBlock(nn.Module):
    def __init__(self,channels,n_heads,head_dim,dropout,context_dim=None):
        super().__init__()
        self.in_channel = channels
        inner_dim = int(channels *  4)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        
        # 2 attention  layers and  a feed forward,
        self.attn1 = CrossAttn(query_dim=channels,heads=n_heads,dim_head=head_dim,dropout=dropout) #self attn (context=None)
        self.attn2 = CrossAttn(query_dim=channels,context_dim=context_dim,heads=n_heads,
                               dim_head=head_dim,dropout=dropout) # crossatn with  context
        
        self.ff  = nn.Sequential(
            nn.Linear(channels,inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim,channels)
        )
    def forward(self,x,context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x),context)  + x  #passing context into forward method 
        x = self.ff(self.norm3(x)) + x
        return x
    
class SpatialTransformer(nn.Module):
    def __init__(self,
                channels,
                n_heads,
                head_dim,
                context_dim=None,
                depth = 1,
                dropout=0.1,               
                ):
        super().__init__()
        self.in_channel = channels
        
        inner_dim = int(n_heads * head_dim)
        self.norm = Normalize(channels)
        # 1x 1conv
        self.proj_in =  nn.Conv2d(
                                  channels,
                                  inner_dim,
                                  kernel_size=1 ,
                                  stride=1,
                                  padding=0 
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(channels,n_heads,head_dim,dropout,context_dim) for _ in range(depth)]
            )
        self.proj_out =  nn.Conv2d(
                                  inner_dim,
                                  channels,
                                  kernel_size=1  
        )
    
        
    def forward(self,x,context_matrix=None):
        # x :(b,c,h,w)
        # context_matrix : (b,seq,dim)
        b,c,h,w = x.shape
        
        x_in = x
        res = self.norm(x_in)
        res = self.proj_in(res)#(b,c,h,w)
        #1D Sequence Format (Flattening)
        #res =  res.permute(0,2,3,1).reshape(b,h*w,c)
        res = rearrange(res,'b c h w -> b (h w) c')#(Batch, Pixels, Channels)
        for block in self.transformer_blocks:
            res = block(res,context_matrix) # (batch,pixels,channels)
        res = res.view(b,h,w,c).permute(0,3,1,2) 
        res = self.proj_out(res) + x # (b,c,h,w)
        return res                 