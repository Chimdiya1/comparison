import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()

        # -------- Encoder (Down) --------
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.d5 = DoubleConv(512, 1024)

        self.pool = nn.MaxPool2d(2)

        # -------- Decoder (Up) --------
        self.u4 = DoubleConv(1024 + 512, 512)
        self.u3 = DoubleConv(512 + 256, 256)
        self.u2 = DoubleConv(256 + 128, 128)
        self.u1 = DoubleConv(128 + 64, 64)

        # -------- Output --------
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):

        # Encoder
        c1 = self.d1(x)
        p1 = self.pool(c1)

        c2 = self.d2(p1)
        p2 = self.pool(c2)

        c3 = self.d3(p2)
        p3 = self.pool(c3)

        c4 = self.d4(p3)
        p4 = self.pool(c4)

        bottleneck = self.d5(p4)

        # Decoder
        up4 = nn.functional.interpolate(bottleneck, scale_factor=2, mode="bilinear")
        cat4 = torch.cat([up4, c4], dim=1)
        c6 = self.u4(cat4)

        up3 = nn.functional.interpolate(c6, scale_factor=2, mode="bilinear")
        cat3 = torch.cat([up3, c3], dim=1)
        c7 = self.u3(cat3)

        up2 = nn.functional.interpolate(c7, scale_factor=2, mode="bilinear")
        cat2 = torch.cat([up2, c2], dim=1)
        c8 = self.u2(cat2)

        up1 = nn.functional.interpolate(c8, scale_factor=2, mode="bilinear")
        cat1 = torch.cat([up1, c1], dim=1)
        c9 = self.u1(cat1)

        return self.out(c9)
