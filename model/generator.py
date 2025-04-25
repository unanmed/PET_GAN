import torch
import torch.nn as nn
import torch.nn.functional as F
from .feat import FeatExtractor

def print_memory(tag=""):
    print(f"{tag} | 当前显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, 最大显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
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
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        if self.drop:
            x = self.drop(x)
        x = self.pool(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.out_ch = out_ch  # 添加out_ch属性
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.conv = ConvBlock(out_ch*2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        return self.conv(x)
    
class BottleneckTransformer(nn.Module):
    def __init__(self, in_ch, hidden_dim=1024):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 1)
        )
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_dim, 
                nhead=8, 
                dim_feedforward=hidden_dim*2,
                batch_first=True
            ),
            num_layers=6
        )
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dim, in_ch, 1),
            nn.InstanceNorm2d(in_ch),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.input(x)
        x = x.view(B, H * W, C)
        x = self.trans(x)
        x = x.view(B, C, H, W)
        x = self.output(x)
        return x

class InputConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(channels[1]),
            nn.GELU(),
            
            nn.Conv2d(channels[1], channels[2], 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(channels[2]),
            nn.GELU(),
            
            nn.Conv2d(channels[2], channels[3], 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(channels[3]),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv_layers(x)
    
class PETUNet(nn.Module):
    def __init__(self, base_ch=64, in_ch=1, out_ch=1):
        super(PETUNet, self).__init__()

        self.in_conv = InputConv([in_ch, 32, 48, base_ch])
        self.down1 = Encoder(base_ch, base_ch * 2, dropout=0.2)
        self.down2 = Encoder(base_ch * 2, base_ch * 4, dropout=0.2)
        self.down3 = Encoder(base_ch * 4, base_ch * 8, dropout=0.2)
        self.down4 = Encoder(base_ch * 8, base_ch * 16, dropout=0.2)

        self.bottleneck = BottleneckTransformer(base_ch * 16, hidden_dim=1024)
        self.fusion = ConvBlock(base_ch * 32, base_ch * 16)

        # 上采样层
        self.up1 = Decoder(base_ch * 16, base_ch * 8)
        self.up2 = Decoder(base_ch * 8, base_ch * 4)
        self.up3 = Decoder(base_ch * 4, base_ch * 2)
        self.up4 = Decoder(base_ch * 2, base_ch)
        self.out_conv = OutConv(base_ch, out_ch)

    def forward(self, x):
        # 编码器路径
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 瓶颈层
        x5_cnn = self.down4(x4)
        x5_trans = self.bottleneck(x5_cnn)
        x5 = self.fusion(torch.cat([x5_cnn, x5_trans], dim=1))

        # 解码器路径并调整通道
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x = self.out_conv(x)
        return x
    
class PETModel(nn.Module):
    def __init__(self, in_ch=32, base_ch=64, out_ch=1, img_count=164):
        super().__init__()
        self.input = ConvBlock(1, in_ch // 2)
        self.ext = FeatExtractor(img_count=img_count, out_ch=in_ch // 2)
        self.unet = PETUNet(in_ch=in_ch, base_ch=base_ch, out_ch=out_ch)
        
    def forward(self, x, glob):
        x = self.input(x)
        x_ext: torch.Tensor = self.ext(glob)
        x = torch.cat([x, x_ext], dim=1)
        x = self.unet(x)
        return F.tanh(x)

# 检查显存占用
if __name__ == "__main__":
    x = torch.rand((1, 8, 256, 256)).cuda()
    glob = torch.rand((1, 164, 256, 256)).cuda()
    model = PETModel(in_ch=16, out_ch=1).cuda()
    print_memory("初始化后")
    output = model(x)
    print_memory("前向传播后")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量：{sum(p.numel() for p in model.parameters())}")
