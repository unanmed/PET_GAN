import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class PETCritic(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(1, base_ch, 3, padding=1)), # 256*256
            nn.MaxPool2d(2), # 128*128
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(base_ch, base_ch*2, 3, padding=1)), # 128*128
            nn.MaxPool2d(2), # 64*64
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1)), # 64*64
            nn.MaxPool2d(4), # 16*16
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(base_ch*4*16*16, 1)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
