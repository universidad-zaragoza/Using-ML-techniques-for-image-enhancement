"""
Archivo de definición del modelo del discriminador.
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Lambda, LeakyReLU, Flatten, Dense

def discriminator_block(input, num_filters, strides=1, batch_norm=True):
    # Capa de convolución del bloque discriminatorio
    db = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(input)
    # Función de activación LeakyReLU
    db = LeakyReLU(alpha=0.2)(db)
    if batch_norm:
        # Capa de Batch Normalization
        db = BatchNormalization(momentum=0.8)(db)
    return db

def normalize(x):
    # Normaliza imágenes a [-1, 1]
    return x / 127.5 - 1

def build_discriminator(hr_crop_size):
    # Se define el tipo de entrada
    input_layer = Input(shape=(hr_crop_size, hr_crop_size, 1))
    
    # Se aplica la normalización a las imágenes de entrada
    x = Lambda(normalize)(input_layer)
    
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
