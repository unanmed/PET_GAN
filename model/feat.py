import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import ChannelAttention

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class FeatExtractor(nn.Module):
    def __init__(self, img_count=164, base_ch=256, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(img_count, base_ch),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
            
            nn.Conv2d(base_ch, out_ch, 1),
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.final_conv(x)
        return x
    