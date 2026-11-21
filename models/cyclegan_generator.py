"""RGB(3) -> thermal(1) ResNet-9 generator with CycleGAN G_A state keys.

Architecture reference:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
Only inference architecture is implemented here; no GAN training dependencies.
"""

from torch import nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, images):
        return images + self.conv_block(images)


class CycleGANGenerator(nn.Module):
    """ResNet-9, instance norm (no affine/running stats), no dropout."""

    def __init__(self, ngf: int = 64):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7),
                  nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True)]
        channels = ngf
        for _ in range(2):
            layers.extend([nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(channels * 2), nn.ReLU(inplace=True)])
            channels *= 2
        layers.extend(_ResidualBlock(channels) for _ in range(9))
        for _ in range(2):
            layers.extend([nn.ConvTranspose2d(channels, channels // 2, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(channels // 2), nn.ReLU(inplace=True)])
            channels //= 2
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(ngf, 1, 7), nn.Tanh()])
        self.model = nn.Sequential(*layers)

    def forward(self, images):
        return self.model(images)
