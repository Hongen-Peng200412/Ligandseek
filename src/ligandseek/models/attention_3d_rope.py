import torch
import einops
from torch import nn
import random
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops.layers.torch import Rearrange
import contextlib

def get_lattice_meshgrid_np(shape_size, no_shift=False):
\
                                                   
                                   
                              
                                                        
                                          
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
                                                          
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels, freq_inv=100):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        super().__init__()
                                              
        self.org_channels = channels

                                                    
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels

                                                                                                                         
        inv_freq = 1.0 / (freq_inv ** (torch.arange(0, channels, 2).float() / channels))
                                               
                                                         
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
                                   
                                 
                                   
                                         
                                                                                     
                                                     
                                                        
         
                      
                                                             
                                                     
         
                           
                                                                       
                                                                                             
          
               
                                     
                                                          
                                                                 
        sin_inp_x = torch.einsum("...i, j -> ...ij", tensor, self.inv_freq.to(tensor.device))[...,None]                           


                                   
                                                  
                                   
                                                    
                                                    
                                                                       
                                                                     
                                                                                                                                             
         
                     
                                                          
                                                                       
                                                                         
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).flatten(-2)
        return emb_x




def ThreeD_Rope(q, k, pos_emb, edge_index=None):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       

                                
                       
                                
                                                                                                                                                                      
                                                                                                                
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)                    

                                                         
                                                                            
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)                




                                
                       
                                
    if edge_index is None:
                                                                                                                                                                                               
                                                                                               
                                          
                                                
                                                 
                                                                      
                                                                                                                                                                                                                                                
        q_new = (
            q * cos_pos[..., None, :]                             
            + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[..., None, :]
        )
        k_new = (
            k * cos_pos[..., None, :]             
            + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[..., None, :]
        )



    else:
                                    
                                                        
                                    
                                                 
        q_new = (
            q * cos_pos[..., None, :]
            + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[..., None, :]
        )

                                                                     
             
                              
                                    
                                                                            
                                                                                                        
        k_new = (
            k * cos_pos[edge_index][..., None, :]
            + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[edge_index][..., None, :]
        )

                                                                                                                                                 

                           
    return q_new, k_new





class Transition(nn.Module):
\
\
\
\
\
\
\
\
\
       
    def __init__(self, in_features:int, norm:nn.Module, n:int=3):
        super().__init__()
        self.norm = norm(in_features)
        self.w1 = nn.Linear(in_features, in_features * n, bias=False)
        self.w2 = nn.Linear(in_features, in_features * n, bias=False)
        self.w3 = nn.Linear(in_features * n, in_features, bias=False)
        self.short = nn.Identity()

    def forward(self, x):
                                       
                                                                
        y = self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))
                  
        y = self.norm(y + self.short(x))
        return y






class AttentionWith3DRoPE(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    def __init__(self, in_features:int, attention_heads:int, attention_features:int):
        super(AttentionWith3DRoPE, self).__init__()
        self.ifz = in_features                                    
        self.ahz = attention_heads                       
        self.afz = attention_features                                                                                                                                                                                          
                                             
        self.attention_scale = np.sqrt(self.afz)

                                                             
                           
        self.q = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.k = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )

                                            
        self.back = nn.Sequential(
            Rearrange("B L ahz afz -> B L (ahz afz)", ahz=self.ahz, afz=self.afz),
            nn.Linear(self.ahz * self.afz, self.ifz, bias=False)
        )

                                                                  
                                                            
        self.pos_encoding = SinusoidalPositionalEncoding(channels=attention_features // 3)

                                          
        self.norm1 = nn.LayerNorm(in_features)
        self.transition1 = Transition(in_features, nn.LayerNorm)

    def forward(self, x):
                            
        B, C, H, D, W = x.shape
                                                                                      
        pos_grid = torch.from_numpy(get_lattice_meshgrid_np((H, D, W), no_shift=True)).float().to(x.device)[None].repeat(B, 1, 1, 1, 1)
        pos_vector = einops.rearrange(pos_grid, "B H D W C -> B (H D W) C") * 1.5                                                                 
                                         
                                                                                   
                                                                                                                        
                                                                                                                                                                                                                        
        pos_emb = self.pos_encoding(pos_vector)
        pos_emb = pos_emb.flatten(-2)
        

                                        
        x_vec = einops.rearrange(x, "B C H D W -> B (H D W) C")                        

                                                                                                              
        query = self.q(x_vec)                       
        key = self.k(x_vec)                       
        value = self.v(x_vec)                     

                                     
                                                                           
        query, key = ThreeD_Rope(query, key, pos_emb)

                                                           
                                                                                                                                                                                  
                                                                                          
        attention_scores = torch.einsum('blai,bkai->blka', query, key) / self.attention_scale

                                                            
        attention_weights = attention_scores.softmax(dim=-2)              

                                                                                                                                                                                                                                      
        out = torch.einsum('blka,bkai->blai', attention_weights, value)

                                                         
        out = self.norm1(x_vec + self.back(out))

                                                       
        out = self.transition1(out)

                                               
        return einops.rearrange(out, "B (H D W) C -> B C H D W", B=B, C=C, H=H, D=D, W=W)
