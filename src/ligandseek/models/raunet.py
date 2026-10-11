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
   
import torch
import einops
from torch import nn
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops.layers.torch import Rearrange
import contextlib
from ligandseek.models.attention_3d_rope import *


class Bottleneck(nn.Module):
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
       
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups=1,
        activation_class=nn.ReLU,
        conv_class=nn.Conv3d,
        affine=False,
        checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(
            in_planes, planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.norm2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(
            planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)

                                         
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv3d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=groups,
            )
                                   
        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal


    def forward_normal(self, x):
\
\
           
        out = self.activation_fn(self.norm1(self.conv1(x)))
        out = self.activation_fn(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

    def forward_checkpoint(self, x):
\
\
           
        return torch_checkpoint(self.forward_normal, x, preserve_rng_state=False)





class ConvBuildingBlock(nn.Module):
\
\
       
    def __init__(self, in_channels:int, out_channels:int, activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels,affine=True),
            self.activate_function,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        
        self.shortcut_conv = nn.Identity()
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )
    
    def forward(self,x:torch.Tensor):
        return self.activate_function(self.conv1(x) + self.shortcut_conv(x))




class ShortConv(nn.Module):
\
\
       
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels,affine=True)
        self.relu1 = nn.ReLU()
    
    def forward(self,x:torch.Tensor):
        y = self.norm1(self.conv1(x))
        y = self.relu1(y)
        return y




class ShortConvAdd(nn.Module):
\
\
\
\
\
\
       
    def __init__(self, input_channels, output_channels: int):
        super().__init__()
        self.output_channels = output_channels
        if input_channels is None:
                                                       
            self.conv1 = nn.LazyConv3d(output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=False)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.relu1 = nn.ELU()
    
    def forward(self,x0,x1):
        y = self.norm1(self.conv1(x0))
        y = self.relu1( y + self.norm2(x1))
        return y





class Res2NetBlock(nn.Module):
\
\
\
\
\
       
    def __init__(self, in_channels, out_channels, stride=1, scale=4,activate_class:nn.Module=nn.ReLU):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, out_channels*self.scale, kernel_size=1, stride=1, padding=0, bias=False), nn.InstanceNorm3d(out_channels*self.scale, affine=True))
        self.norm1 = nn.InstanceNorm3d(out_channels*self.scale, affine=True)
        self.conv_list = nn.ModuleList([nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) for _ in range(self.scale - 1)])
        self.activate_class = activate_class()
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels*self.scale, out_channels, 1, 1, 0, bias=False), nn.InstanceNorm3d(out_channels, affine=True))
        
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=True
            )
    
    def forward(self, x):
                                                                           
        x_list = self.activate_class(self.conv1(x)).chunk(self.scale,dim=1)                          
                                                  
        y_list = []
        for ii,xi in enumerate(x_list):
            if ii == 0:
                y_list.append(xi)
            elif ii == 1:
                y_list.append(self.conv_list[ii-1](xi))
            else:
                y_list.append(self.conv_list[ii-1](xi+y_list[-1]))
        
        y = self.conv2(self.activate_class(self.norm1(  torch.cat(y_list,dim=1)  )))
        y = self.activate_class(y+self.shortcut_conv(x))
        return y
    




class AttentionGate(nn.Module):
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
       
    def __init__(self, down_features:int, up_features:int, out_features:int, attention_features:int=64, attention_heads:int=8):
        super(AttentionGate, self).__init__()
        self.dfz = down_features
        self.ufz = up_features
        self.ofz = out_features
        self.afz = attention_features
        self.ahz = attention_heads
        
                                               
        self.conv_q = nn.Sequential(nn.Conv3d(
            in_channels=self.ufz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        
                                                 
        self.conv_k = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        
                                                                      
        self.conv_v = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.ufz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.ufz,affine=True))
        
                                                                        
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.afz,
                out_channels=self.ahz,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
                                         
        self.conv_back = ConvBuildingBlock(self.ufz, self.ofz)
    
    def forward(self,us,ds):
\
\
\
\
\
\
\
           
        ds_shape = ds.shape
        D, H, W = ds_shape[2:]
                              
        upsampled = nn.functional.interpolate(input=us, size=(D, H, W), mode='trilinear', align_corners=True)
        query = self.conv_q(upsampled)
        key = self.conv_k(ds)
        value = self.conv_v(ds)
        
                                              
                                                  
        value = einops.rearrange(value, "N (afz ahz) d h w -> N afz ahz d h w", ahz=self.ahz)
        
                                                       
                              
        gate = self.gate(query+key)               
        
                                            
                             
        out = value*gate[:,None]
        
                                               
                      
        out = einops.rearrange(out,"N afz ahz d h w -> N (afz ahz) d h w", ahz=self.ahz)
        
                                           
        return self.conv_back(self.relu(out+upsampled))


















class SimpleUnet(nn.Module):
\
\
\
\
\
\
       
    def __init__(
        self,
        in_channels: int = None,
        out_channels: int = 1,
        planes: list = (64, 256, 256, 256, 256, 256, 128, 64, 64),
        gradient_checkpoint: bool = False,
    ):
        super(SimpleUnet, self).__init__()
                                                                                     
        enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0 = planes
        self._planes_for_reinit = planes

                                                          
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gradient_checkpoint = bool(gradient_checkpoint)
        
                                                                          
        self.shortconvadd = ShortConvAdd(input_channels=in_channels, output_channels=enc0)                   
        self.shortconv1 = ShortConv(enc0, enc0 * 4)                                   
        
                               
                                                            
        self.downsample1 = Bottleneck(enc0 * 4, enc1 // 4, stride=2, affine=True)
                                                          
        self.downsample2 = Bottleneck(enc1, enc2 // 4, stride=2, affine=True)
                                                          
        self.downsample3 = Bottleneck(enc2, enc3 // 4, stride=2, affine=True)
                                                                
        self.downsample4 = Bottleneck(enc3, bottleneck // 4, stride=2, affine=True)
        
                                      
                                               
        self.A_block = nn.Sequential(*[AttentionWith3DRoPE(bottleneck, 8, bottleneck // 8 * 6) for _ in range(4)])
        

                               
                                                                                          
        self.main1 = self.main_layer(dec3, 3, 4)
        
                                                                       
        self.attn2 = AttentionGate(enc2, dec3, dec2)
                                              
        self.main2 = self.main_layer(dec2, 4, 4)
        
                                                                       
        self.attn3 = AttentionGate(enc1, dec2, dec1)
                                              
        self.main3 = self.main_layer(dec1, 4, 4)
        
                                                                        
        self.attn4 = AttentionGate(enc0 * 4, dec1, dec0)
                                              
        self.main4 = self.main_layer(dec0, 4, 4)
        
                            
        mid_ch = max(32, dec0 // 2)
                                        
        self.conv_end_3 = nn.Conv3d(dec0, mid_ch, kernel_size=3, stride=1, padding=1)
                                        
        self.conv_end_5 = nn.Conv3d(dec0, mid_ch, kernel_size=5, stride=1, padding=2)
                                        
        self.conv_end_7 = nn.Conv3d(dec0, mid_ch, kernel_size=7, stride=1, padding=3)
                              
        self.relu1 = nn.ReLU()
                                     
        self.conv_end = nn.Conv3d(in_channels=mid_ch * 3, out_channels=out_channels, padding=1, kernel_size=3)

    def set_input_channels(self, in_channels: int):                                                   
\
\
           
        self.in_channels = in_channels
                            
        current_out = self.shortconvadd.norm2.num_features             
        device = next(self.parameters()).device                           
        self.shortconvadd = ShortConvAdd(input_channels=in_channels, output_channels=current_out)
        self.shortconvadd.to(device)                                     

    def _checkpoint_call(self, fn, *args):
\
\
           
        if not (self.gradient_checkpoint and self.training and torch.is_grad_enabled()):
            return fn(*args)
        try:
            return torch_checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        except TypeError:
                                                                                    
            return torch_checkpoint(fn, *args, preserve_rng_state=False)
    
    def main_layer(self,input_channels,expansion,num_layers):
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
           
                                       
        layer=[]
        for i in range(num_layers):
            layer.append(Res2NetBlock(input_channels,input_channels,scale=expansion))
        return nn.Sequential(*layer)
    
    def upsample_add(self,f, g):
\
\
\
\
\
\
\
\
\
           
        g_shape = g.shape
        D, H, W = g_shape[2:]
        upsampled = nn.functional.interpolate(input=f, size=(D, H, W), mode='trilinear', align_corners=True)
        return (g + upsampled)
    
                     
    def forward(self,voxel_grid,run_iters:int=3):
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
           
        B, C, D, H, W = voxel_grid.shape
        
                                                                                    
        voxel_recycle = torch.zeros((B, 64, D, H, W), device=voxel_grid.device, dtype=voxel_grid.dtype)

        for run_iter in range(run_iters):
            not_last_iter = (run_iter != (run_iters - 1))
            
                                            
            with torch.no_grad() if not_last_iter else contextlib.nullcontext():
                                                                   
                fused_input = self.shortconvadd(voxel_grid,voxel_recycle)
                
                                   
                                                        
                ds_0 = self.shortconv1(fused_input)
                                                              
                ds_1 = self._checkpoint_call(self.downsample1, ds_0)
                                                              
                ds_2 = self._checkpoint_call(self.downsample2, ds_1)
                                                              
                ds_3 = self._checkpoint_call(self.downsample3, ds_2)
                                                                      
                ds_4 = self._checkpoint_call(self.downsample4, ds_3)
                
                                             
                                                                     
                c4 = self._checkpoint_call(self.A_block, ds_4)
                
                                   
                                                                 
                c3 = self._checkpoint_call(self.main1, self.upsample_add(c4, ds_3))
                                                                  
                c2 = self._checkpoint_call(self.main2, self._checkpoint_call(self.attn2, c3, ds_2))
                                                                 
                c1 = self._checkpoint_call(self.main3, self._checkpoint_call(self.attn3, c2, ds_1))
                                                           
                c0 = self._checkpoint_call(self.main4, self._checkpoint_call(self.attn4, c1, ds_0))
                
                                   
                                                      
                f3 = self.conv_end_3(c0)
                                                      
                f5 = self.conv_end_5(c0)
                                                      
                f7 = self.conv_end_7(c0)
                
                                                                      
                fused_multiscale = self.relu1(torch.cat((f3, f5, f7), dim=1))
                                                                      
                final_feature = self.conv_end(fused_multiscale)
                
                                                                      
                voxel_recycle = c0.detach()
        return final_feature





