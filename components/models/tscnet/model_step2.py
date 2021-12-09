import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    '''
        This block comes from https://github.com/Harshubh-Meherishi/Residual-Dense-Networks
    '''
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)  # Modified here

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    '''
        This block comes from https://github.com/Harshubh-Meherishi/Residual-Dense-Networks
    '''
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        # self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=(1, 1))
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, max(growth_rate, in_channels), kernel_size=(1, 1))  # Modified

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    '''
        This block comes from https://github.com/Harshubh-Meherishi/Residual-Dense-Networks
        The range of scale_factor is expanded from [2-4] to [1-4]
    '''
    def __init__(self, scale_factor, in_channels, out_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        self.in_channels = in_channels

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=3 // 2)
        # self.sfe2 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=3 // 2)
        self.sfe2 = nn.Conv2d(in_channels, self.G0, kernel_size=(3, 3), padding=3 // 2)  # Modified here

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(max(self.G, self.G0), self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=(1, 1)),
            # nn.Conv2d(self.G0, self.G0, kernel_size=(3, 3), padding=3 // 2),
            nn.Conv2d(self.G0, self.in_channels, kernel_size=(3, 3), padding=3 // 2),  # Midified here
        )

        # up-sampling
        assert 1 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                # self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=(3, 3), padding=3 // 2),
                self.upscale.extend([nn.Conv2d(self.in_channels, self.G0 * (2 ** 2), kernel_size=(3, 3), padding=3 // 2),  # Modified here
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                # nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=(3, 3), padding=3 // 2),
                nn.Conv2d(self.in_channels, self.G0 * (scale_factor ** 2), kernel_size=(3, 3), padding=3 // 2),  # Modified here
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, out_channels, kernel_size=(3, 3), padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    '''
    This class comes from https://github.com/leftthomas/SRGAN
    '''
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2),
            #
            # nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2),
            #
            # nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(512, 1024, kernel_size=(1, 1)),
            # nn.LeakyReLU(0.2),
            # nn.Conv2d(1024, 1, kernel_size=(1, 1))

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
    """
    Base class for Step2 of TSCNet proposed in
    Lu, Z., et al. (2021). "Two-Stage Self-Supervised Cycle-Consistency Network for Reconstruction of Thin-Slice MR Images." arXiv e-prints: arXiv:2106.15395.
    """

    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        assert in_channels % 2 == 0, f'in_channels % 2 should be 0, got in_channels={in_channels} instead'
        self.in_channels = in_channels // 2
        self.out_channels = out_channels

        self.rdn1 = RDN(scale_factor=1, in_channels=self.in_channels, out_channels=self.in_channels, num_features=self.in_channels, growth_rate=4, num_blocks=8, num_layers=16)
        self.rdn2 = RDN(scale_factor=1, in_channels=self.in_channels, out_channels=self.in_channels, num_features=self.in_channels, growth_rate=4, num_blocks=8, num_layers=16)

        self.rdn3 = RDN(scale_factor=1, in_channels=self.in_channels*2, out_channels=self.out_channels, num_features=self.in_channels, growth_rate=4, num_blocks=3, num_layers=3)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)  # split channels
        x1 = self.rdn1(x1)
        x2 = self.rdn2(x2)

        output = self.rdn3(torch.cat([x1, x2], dim=1))
        return output


class AbstractTSCNetStep2(nn.Module):
    """
    Base class for Step2 of TSCNet proposed in
    Lu, Z., et al. (2021). "Two-Stage Self-Supervised Cycle-Consistency Network for Reconstruction of Thin-Slice MR Images." arXiv e-prints: arXiv:2106.15395.
    """

    def __init__(self, in_channels, out_channels):
        super(AbstractTSCNetStep2, self).__init__()

        assert in_channels % 2 == 0, f'in_channels % 2 should be 0, got in_channels={in_channels} instead'
        self.generator = Generator(in_channels=in_channels, out_channels=out_channels)
        self.discriminator = Discriminator(in_channels=out_channels)

    def forward(self, x):
        return self.generator(x)


class TSCNetStep2(AbstractTSCNetStep2):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels):
        super(TSCNetStep2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
        )




