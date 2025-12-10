import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMBlock(nn.Module):
    """Two-conv residual block with GroupNorm and FiLM conditioning."""
    def __init__(self, in_ch: int, out_ch: int, z_dim: int, groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.aff1  = nn.Linear(z_dim, 2 * out_ch)  # produces gamma, beta

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.aff2  = nn.Linear(z_dim, 2 * out_ch)

        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

        # Initialize FiLM to identity (gamma=1, beta=0) so delta starts near zero
        nn.init.zeros_(self.aff1.weight); nn.init.zeros_(self.aff1.bias)
        nn.init.zeros_(self.aff2.weight); nn.init.zeros_(self.aff2.bias)

    def forward(self, x, z):
        g1, b1 = self.aff1(z).chunk(2, dim=-1); g2, b2 = self.aff2(z).chunk(2, dim=-1)
        g1 = g1.unsqueeze(-1).unsqueeze(-1); b1 = b1.unsqueeze(-1).unsqueeze(-1)
        g2 = g2.unsqueeze(-1).unsqueeze(-1); b2 = b2.unsqueeze(-1).unsqueeze(-1)

        y = self.conv1(x); y = self.gn1(y); y = g1 * y + b1; y = self.act(y)
        y = self.conv2(y); y = self.gn2(y); y = g2 * y + b2; y = self.act(y)
        return y + self.skip(x)


class Down(nn.Module):
    """Stride-2 downsampling followed by a FiLM block."""
    def __init__(self, in_ch: int, out_ch: int, z_dim: int):
        super().__init__()
        self.conv_s2 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.block   = FiLMBlock(out_ch, out_ch, z_dim)
    def forward(self, x, z):
        x = self.conv_s2(x)
        return self.block(x, z)


class Up(nn.Module):
    """Nearest-neighbor upsampling + skip connection + FiLM block."""
    def __init__(self, in_ch: int, out_ch: int, z_dim: int):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1)
        self.block  = FiLMBlock(out_ch, out_ch, z_dim)
    def forward(self, x, skip, z):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.block(x, z)


class ConditionalUNetDelta(nn.Module):
    """Lightweight U-Net that outputs delta images with FiLM conditioning by z."""
    def __init__(self, in_ch: int, out_ch: int, z_dim: int, base: int = 64, depth: int = 2):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 2

        self.enc0 = FiLMBlock(in_ch, c1, z_dim)
        self.down1 = Down(c1, c2, z_dim)
        self.down2 = Down(c2, c3, z_dim) if depth >= 2 else None

        self.mid = FiLMBlock(c3 if depth >= 2 else c2, c3, z_dim)

        self.up2 = Up(c3 + (c2 if depth >= 2 else 0), c2, z_dim) if depth >= 2 else None
        self.up1 = Up(c2 + c1, c1, z_dim)

        self.head = nn.Conv2d(c1, out_ch, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, z):
        e0 = self.enc0(x, z)          # (B, c1, H, W)
        e1 = self.down1(e0, z)        # (B, c2, H/2, W/2)
        if self.down2 is not None:
            e2 = self.down2(e1, z)    # (B, c3, H/4, W/4)
            m  = self.mid(e2, z)
            u2 = self.up2(m, e1, z)   # (B, c2, H/2, W/2)
            u1 = self.up1(u2, e0, z)  # (B, c1, H, W)
        else:
            m  = self.mid(e1, z)
            u1 = self.up1(m, e0, z)
        return self.head(u1)          # (B, out_ch, H, W)