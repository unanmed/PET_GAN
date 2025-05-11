import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionEncoder(nn.Module):
    def __init__(self, in_ch=164, out_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1, padding_mode='replicate'), # 256x256
            nn.InstanceNorm2d(256),
            nn.GELU(),
            
            nn.MaxPool2d(4),
            nn.Conv2d(256, 512, 3, padding=1, padding_mode='replicate'), # 64x64
            nn.InstanceNorm2d(512),
            nn.GELU(),
            
            nn.MaxPool2d(4),
            nn.Conv2d(512, 512, 3, padding=1, padding_mode='replicate'), # 16x16
            nn.InstanceNorm2d(1024),
            nn.GELU(),
            
            nn.MaxPool2d(4),
            nn.Conv2d(512, 1024, 3, padding=1, padding_mode='replicate'), # 4x4
            nn.InstanceNorm2d(1024),
            nn.GELU(),
        )
        self.out = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(1024, 1024, 1),
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, out_dim)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.out(x)
        x = x.squeeze(3).squeeze(2)
        x = self.fc(x)
        return x
    
class ConditionInjector(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.beta_fc = nn.Sequential(
            nn.Linear(cond_dim, out_dim)
        )
        self.gamma_fc = nn.Sequential(
            nn.Linear(cond_dim, out_dim)
        )

    def forward(self, x, cond):
        beta = self.beta_fc(cond).unsqueeze(2).unsqueeze(3)
        gamma = self.gamma_fc(cond).unsqueeze(2).unsqueeze(3)
        x = x * gamma + beta
        return x
