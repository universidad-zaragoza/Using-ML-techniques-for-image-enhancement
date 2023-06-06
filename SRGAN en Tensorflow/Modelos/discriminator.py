# coding=utf-8
# discriminator.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu and Pablo Doñate.
#
# Using Machine Learning techniques for image enhancement.
# This file has been created by jlaihong and modified by Pablo Doñate Navarro (800710@unizar.es).

"""
    Este archivo define la función para construir el modelo del discriminador de la red SRGAN.
    El discriminador es una red convolucional que clasifica las imágenes como reales o falsas, y consta de 
    ocho bloques de capas convolucionales, una capa densa y una capa de salida.
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, LeakyReLU, Flatten, Dense

def discriminator_block(input, num_filters, strides=1, batch_norm=True):
    db = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(input)
    db = LeakyReLU(alpha=0.2)(db)
    if batch_norm:
        db = BatchNormalization(momentum=0.8)(db)
    return db

def normaliza(x):
    # Normaliza imágenes a [-1, 1].
    return x / 127.5 - 1

def build_discriminator(hr_crop_size):
    # Se define el tipo de entrada
    input_layer = Input(shape=(hr_crop_size, hr_crop_size, 1))
    
    x = Lambda(normaliza)(input_layer)
    
    # Se define la primera capa asociada al primer bloque discriminatorio.
    # Este bloque no cuenta con Batch Normalization.
    d1 = discriminator_block(x, 64, batch_norm=False)
    
    # Se definen el resto de bloques discriminatorios.
    d2 = discriminator_block(d1, 64, strides=2)
    d3 = discriminator_block(d2, 128)
    d4 = discriminator_block(d3, 128, strides=2)
    d5 = discriminator_block(d4, 256)
    d6 = discriminator_block(d5, 256, strides=2)
    d7 = discriminator_block(d6, 512)
    d8 = discriminator_block(d7, 512, strides=2)

    x = Flatten()(d8)

    f_layer = Dense(1024)(x)
    f_layer = LeakyReLU(alpha=0.2)(f_layer)
    dis_output = Dense(units=1, activation='sigmoid')(f_layer)

    return Model(inputs=input_layer, outputs=dis_output)