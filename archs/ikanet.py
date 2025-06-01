import torch
from torch import nn
import torch.nn.functional as F
from archs.fastkanconv import FastKANConvLayer
from archs.improvedfastkanconv import ImprovedFastKANConvLayer


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.conv1 = ImprovedFastKANConvLayer(in_channels, out_channels // 2, padding=1, kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        # self.conv2 = ImprovedFastKANConvLayer(out_channels // 2, out_channels, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, padding=1, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_channels)
        self.sa = SpatialAttention()

        # 残差连接
        self.shortcut = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out * self.sa(out)  # 使用空间注意力

        out += identity
        out = self.relu(out)
        return out


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        self.device = device
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, device=self.device)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True, device='cuda'):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, device=device)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, device=device)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = ImprovedFastKANConvLayer(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class IKanet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, device='mps'):
        super(IKanet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.device = device

        self.channels = [64, 128, 256, 512, 1024]

        self.inc = (DoubleConv(n_channels, 64, device=self.device))

        self.down1 = (Down(self.channels[0], self.channels[1], self.device))
        self.down2 = (Down(self.channels[1], self.channels[2], self.device))
        self.down3 = (Down(self.channels[2], self.channels[3], self.device))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.channels[3], self.channels[4] // factor, self.device))
        self.up1 = (Up(self.channels[4], self.channels[3] // factor, bilinear, self.device))
        self.up2 = (Up(self.channels[3], self.channels[2] // factor, bilinear, self.device))
        self.up3 = (Up(self.channels[2], self.channels[1] // factor, bilinear, self.device))
        self.up4 = (Up(self.channels[1], self.channels[0], bilinear, self.device))
        self.outc = (OutConv(self.channels[0], n_classes))
        self.ds1 = nn.Conv2d(self.channels[3], n_classes, 1)
        self.ds2 = nn.Conv2d(self.channels[2], n_classes, 1)
        self.ds3 = nn.Conv2d(self.channels[1], n_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        ds1 = F.interpolate(self.ds1(x), size=x.shape[2:], mode='bilinear')

        x = self.up2(x, x3)
        ds2 = F.interpolate(self.ds2(x), size=x.shape[2:], mode='bilinear')

        x = self.up3(x, x2)
        ds3 = F.interpolate(self.ds3(x), size=x.shape[2:], mode='bilinear')

        x = self.up4(x, x1)
        logits = self.outc(x)

        if self.training:
            return logits, ds1, ds2, ds3
        return logits
