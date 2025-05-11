import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CBAM
from .cond import ConditionInjector, ConditionEncoder

def print_memory(tag=""):
    print(
        f"{tag} | 当前显存: {torch.cuda.memory_allocated('cuda:1') / 1024**2:.2f} MB, "
        f"最大显存: {torch.cuda.max_memory_allocated('cuda:1') / 1024**2:.2f} MB"
    )
    
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
    
class SkipEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, window_size=8, heads=4, dim_head=32):
        super().__init__()
        self.window_size = window_size

        embed_dim = heads * dim_head  # 每头的特征数乘头数 = 嵌入维度

        # 1×1卷积投影到 embed_dim
        self.proj_in = nn.Conv2d(in_ch, embed_dim, kernel_size=1)

        # Multi-head self-attention （局部窗口）
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=heads, batch_first=True)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
        )

        # 1×1卷积投影回 out_ch
        self.proj_out = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 输入卷积
        x_proj = self.proj_in(x)  # [B, embed_dim, H, W]

        # 2. 分割成窗口
        ws = self.window_size
        x_proj = x_proj.view(B, -1, H // ws, ws, W // ws, ws)
        x_proj = x_proj.permute(0, 2, 4, 3, 5, 1)  # [B, H/ws, W/ws, ws, ws, embed_dim]
        x_proj = x_proj.reshape(-1, ws * ws, x_proj.shape[-1])  # [num_windows*B, ws*ws, embed_dim]

        # 3. Attention
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        x_proj = x_proj + attn_out
        x_proj = self.norm1(x_proj)

        # 4. MLP
        mlp_out = self.mlp(x_proj)
        x_proj = x_proj + mlp_out
        x_proj = self.norm2(x_proj)

        # 5. 恢复形状
        x_proj = x_proj.view(B, H // ws, W // ws, ws, ws, -1)
        x_proj = x_proj.permute(0, 5, 1, 3, 2, 4)
        x_proj = x_proj.reshape(B, -1, H, W)  # [B, embed_dim, H, W]

        # 6. 输出卷积
        out = self.proj_out(x_proj)  # [B, out_ch, H, W]

        return out

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, attn=True):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.attn = CBAM(out_ch) if attn else None
        self.pool = nn.MaxPool2d(2)
        self.inject = ConditionInjector(1024, out_ch)
        
    def forward(self, x, cond):
        x = self.conv(x)
        if self.attn:
            x = self.attn(x)
        x = self.pool(x)
        x = self.inject(x, cond)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_ch = out_ch  # 添加out_ch属性
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )
        self.conv = ConvBlock(out_ch*2, out_ch)
        self.inject = ConditionInjector(1024, out_ch)
        # self.skip_enc = SkipEncoder(out_ch, out_ch)
        # self.fusion = nn.Conv2d(out_ch*2, out_ch, 1)

    def forward(self, x, skip, cond):
        x = self.up(x)
        # skip_trans = self.skip_enc(skip)
        # skip = torch.cat([skip, skip_trans], dim=1)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.inject(x, cond)
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
        self.inject = ConditionInjector(1024, channels[3])

    def forward(self, x, cond):
        x = self.conv_layers(x)
        x = self.inject(x, cond)
        return x
    
class PETUNet(nn.Module):
    def __init__(self, base_ch=64, in_ch=1, out_ch=1):
        super(PETUNet, self).__init__()

        self.in_conv = InputConv([in_ch, 32, 48, base_ch])
        self.down1 = Encoder(base_ch, base_ch * 2)
        self.down2 = Encoder(base_ch * 2, base_ch * 4)
        self.down3 = Encoder(base_ch * 4, base_ch * 8)
        self.down4 = Encoder(base_ch * 8, base_ch * 16)

        # 上采样层
        self.up1 = Decoder(base_ch * 16, base_ch * 8)
        self.up2 = Decoder(base_ch * 8, base_ch * 4)
        self.up3 = Decoder(base_ch * 4, base_ch * 2)
        self.up4 = Decoder(base_ch * 2, base_ch)
        self.out_conv = OutConv(base_ch, out_ch)

    def forward(self, x, cond):
        # 编码器路径
        x1 = self.in_conv(x, cond)
        x2 = self.down1(x1, cond)
        x3 = self.down2(x2, cond)
        x4 = self.down3(x3, cond)
        x5 = self.down4(x4, cond)

        # 解码器路径并调整通道
        x = self.up1(x5, x4, cond)
        x = self.up2(x, x3, cond)
        x = self.up3(x, x2, cond)
        x = self.up4(x, x1, cond)
        
        x = self.out_conv(x)
        return x
    
class PETModel(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, out_ch=1, img_count=164):
        super().__init__()
        self.cond_enc = ConditionEncoder(img_count)
        self.unet = PETUNet(in_ch=in_ch, base_ch=base_ch, out_ch=out_ch)
        
    def forward(self, x, glob):
        cond = self.cond_enc(glob)
        x = self.unet(x, cond)
        return F.tanh(x)

# 检查显存占用
if __name__ == "__main__":
    x = torch.rand((1, 1, 256, 256)).cuda(1)
    glob = torch.rand((1, 164, 256, 256)).cuda(1)
    model = PETModel(in_ch=1, out_ch=1).cuda(1)
    print_memory("初始化后")
    output = model(x, glob)
    print_memory("前向传播后")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cond Encoder parameters: {sum(p.numel() for p in model.cond_enc.parameters())}")
    print(f"UNet parameters: {sum(p.numel() for p in model.unet.parameters())}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
