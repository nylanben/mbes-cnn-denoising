import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two Conv3d layers with BatchNorm and ReLU. Used in encoder and bottleneck."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TripleConv(nn.Module):
    """Three Conv3d layers with BatchNorm and ReLU. Used in decoder."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv5(nn.Module):
    """Two Conv3d layers with kernel size 5, BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TripleConv5(nn.Module):
    """Three Conv3d layers with kernel size 5, BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 5, padding=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module): # Model K3
    """
    3D U-Net with kernel size 3.
    Input:  (B, 2, 32, 32, 32)
    Output: (B, 1, 32, 32, 32)
    Requires input spatial dims divisible by 8 (three MaxPool3d layers).
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(2, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 128)

        # Decoder
        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.dec3 = TripleConv(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = TripleConv(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = TripleConv(64, 32)

        # Final voxel prediction (raw logits)
        self.final = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = F.interpolate(u3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)


class UNet3D_k5(nn.Module): # Model K5
    """
    3D U-Net with kernel size 5.
    Input:  (B, 2, 32, 32, 32)
    Output: (B, 1, 32, 32, 32)
    Requires input spatial dims divisible by 8 (three MaxPool3d layers).
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv5(2, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv5(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv5(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv5(128, 128)

        # Decoder
        self.up3 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.dec3 = TripleConv5(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec2 = TripleConv5(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1 = TripleConv5(64, 32)

        # Final voxel prediction (raw logits)
        self.final = nn.Conv3d(32, 1, kernel_size=1)

    def forward(self, x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = F.interpolate(u3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)

class UNet3D_f16(nn.Module): # Model F16
    """3D U-Net with kernel size 3, 16 base filters."""
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(2, 16)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(64, 64)

        self.up3 = nn.ConvTranspose3d(64, 64, 2, stride=2)
        self.dec3 = TripleConv(128, 64)

        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = TripleConv(64, 32)

        self.up1 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec1 = TripleConv(32, 16)

        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = F.interpolate(u3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)


class UNet3D_f64(nn.Module): # Model F64
    """3D U-Net with kernel size 3, 64 base filters."""
    def __init__(self):
        super().__init__()

        self.enc1 = DoubleConv(2, 64)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(256, 256)

        self.up3 = nn.ConvTranspose3d(256, 256, 2, stride=2)
        self.dec3 = TripleConv(512, 256)

        self.up2 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec2 = TripleConv(256, 128)

        self.up1 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec1 = TripleConv(128, 64)

        self.final = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = F.interpolate(u3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.final(d1)