import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ConditionEncoder(nn.Module):
    def __init__(self, in_ch=164, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, padding_mode='replicate'), # 256x256
            nn.GroupNorm(8, 128),
            nn.GELU(),
            
            nn.MaxPool2d(4),
            nn.Conv2d(128, 256, 3, padding=1, padding_mode='replicate'), # 64x64
            nn.GroupNorm(8, 256),
            nn.GELU(),
            
            nn.MaxPool2d(4),
            nn.Conv2d(256, 512, 3, padding=1, padding_mode='replicate'), # 16x16
            nn.GroupNorm(8, 512),
            nn.GELU(),
            
            nn.MaxPool2d(4),
            nn.Conv2d(512, 512, 3, padding=1, padding_mode='replicate'), # 4x4
            nn.GroupNorm(8, 512),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(512, 512, 1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, out_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        x = x.squeeze(3).squeeze(2)
        x = self.fc(x)
        return x

class InputHead(nn.Module):
    def __init__(self, base_ch):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(1, base_ch, 3, padding=1)),
            nn.MaxPool2d(4),
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            
            spectral_norm(nn.Conv2d(base_ch, base_ch, 3, padding=1)),
            nn.GroupNorm(8, base_ch),
            nn.GELU()
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class PETCritic(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.cond_conv = InputHead(base_ch)
        self.target_conv = InputHead(base_ch)
        self.cond_enc = ConditionEncoder()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1)), # 64*64
            nn.MaxPool2d(4), # 16*16
            nn.GroupNorm(8, base_ch*4),
            nn.GELU(),
            
            spectral_norm(nn.Conv2d(base_ch*4, base_ch*8, 3, padding=1)), # 16*16
            nn.MaxPool2d(4), # 4*4
            nn.GroupNorm(8, base_ch*8),
            nn.GELU(),
            
            spectral_norm(nn.Conv2d(base_ch*8, base_ch*8, 3, padding=1)), # 4*4
            nn.MaxPool2d(4), # 1*1
            nn.GroupNorm(8, base_ch*8),
            nn.GELU(),
            
            spectral_norm(nn.Conv2d(base_ch*8, base_ch*8, 1)),
            nn.GroupNorm(8, base_ch*8),
            nn.GELU(),
        )
        self.fc = spectral_norm(nn.Linear(base_ch*8, 1))
        self.proj = spectral_norm(nn.Linear(512, base_ch*8))
        
    def forward(self, pred, origin, glob):
        pred = self.cond_conv(pred)
        origin = self.target_conv(origin)
        cond = self.cond_enc(glob)
        x = torch.cat([pred, origin], dim=1)
        x = self.conv(x)
        x = x.squeeze(3).squeeze(2)
        cond = self.proj(cond)
        proj = torch.sum(x * cond, dim=1, keepdim=True)
        x = self.fc(x) + proj
        return x
