# coding=utf-8
# generator.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu and Pablo Doñate.
#
# Using Machine Learning techniques for image enhancement.
# This file has been created by Erik Linder-Norén (eriklindernoren@gmail.com) and 
# modified by Pablo Doñate Navarro (800710@unizar.es).

"""
    Este archivo define una función para construir el modelo del generador de la red SRGAN.
    El generador es una red convolucional que genera imágenes de alta resolución a partir de imágenes de baja resolución.
    Consta de una capa convolucional inicial, varios bloques residuales, varios bloques de upsample y una capa convolucional final.
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
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16):
        super(Generator, self).__init__()

        # Capa convolucional con función de activación 'PReLU'.
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels=64, kernel_size=9, stride=1, padding='same'), nn.PReLU())

        # Bucle con x bloques residuales
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Último bloque residual sin función de activación
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(64, momentum=0.99))

        # Bucle con x bloques de upsample
        upsampling = []
        for out_features in range(2):
            upsampling += [
                nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding='same'),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Última capa de la red generadora con función de activación tanh.
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=9, stride=1, padding='same'), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out