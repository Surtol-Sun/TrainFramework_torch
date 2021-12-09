import torch
import torch.nn as nn
from components.models.unet.unet_parts import *


class Discriminator(nn.Module):
    '''
    This class comes from https://github.com/leftthomas/SRGAN
    '''
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 256, kernel_size=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=(1, 1))
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Generator, self).__init__()
        self.n_channels = in_channels
        self.n_classes = in_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #self.outc = OutConv(64, n_classes)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out


class UNet_Gan(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet_Gan, self).__init__()
        self.discriminator = Discriminator(in_channels=out_channels)
        self.generator = Generator(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        return self.generator(x)


