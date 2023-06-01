"""
Generador SRGAN.
Este archivo contiene la implementación de la red generadora (generador) para el modelo SRGAN (Super-Resolution Generative Adversarial Network).

"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(in_features, momentum=0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(in_features, momentum=0.8),
        )

    def forward(self, x):
        # Paso hacia adelante del bloque residual
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16):
        super(Generator, self).__init__()

        # Primera capa
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=64, kernel_size=9, stride=1, padding='same'), nn.PReLU())

        # Bloques residuales
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Segunda capa convolucional después de los bloques residuales
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(64, momentum=0.99))

        # Capas de aumento de resolución
        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding='same'),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Capa de salida final
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=9, stride=1, padding='same'), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out
