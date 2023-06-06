# coding=utf-8
# vgg19.py
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
    Este archivo define la red extractora de características VGG-19.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        # Convertir la imagen monocromática en RGB de 3 canales
        img_rgb = torch.cat([img, img, img], dim=1)
        return self.feature_extractor(img_rgb)