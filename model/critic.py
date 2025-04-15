import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False))
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.conv = spectral_norm(nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class InputHead(nn.Module):
    def __init__(self, base_ch):
        super().__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(1, base_ch, 3, padding=1)),
            nn.GroupNorm(8, base_ch),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)),
            nn.GroupNorm(8, base_ch),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class PETCritic(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        self.cond_conv = InputHead(base_ch)
        self.target_conv = InputHead(base_ch)
        # self.attention = CBAM(base_ch*2)
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1)), # 128*128
            nn.MaxPool2d(2), # 64*64
            nn.GroupNorm(8, base_ch*4),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(base_ch*4, base_ch*8, 3, padding=1)), # 64*64
            nn.MaxPool2d(2), # 32*32
            nn.GroupNorm(8, base_ch*8),
            nn.LeakyReLU(0.2),
            
            spectral_norm(nn.Conv2d(base_ch*8, base_ch*1, 3, padding=1)), # 32*32
            nn.MaxPool2d(2), # 16*16
            nn.GroupNorm(8, base_ch),
            nn.LeakyReLU(0.2),
            
            # spectral_norm(nn.Conv2d(base_ch*16, base_ch*1, 1)), # 16*16
            # nn.InstanceNorm2d(base_ch),
            # nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(base_ch*16*16, 1))
        )
        
    def forward(self, pred, origin):
        pred = self.cond_conv(pred)
        origin = self.target_conv(origin)
        x = torch.cat([pred, origin], dim=1)
        # x = self.attention(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
