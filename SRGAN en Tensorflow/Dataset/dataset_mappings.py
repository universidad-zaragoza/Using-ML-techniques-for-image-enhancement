# coding=utf-8
# dataset_mappings.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu and Pablo Doñate.
#
# Using Machine Learning techniques for image enhancement.
# This file has been created by jlaihong and modified by Pablo Doñate Navarro (800710@unizar.es).

"""
    Este archivo define funciones para aplicar transformaciones de mapeo a las imágenes de los datasets.
    Las transformaciones incluyen recortes aleatorios, giros aleatorios y rotaciones aleatorias.
"""

import tensorflow as tf

def random_crop(lr_img, hr_img, hr_crop_size, scale):
    # Se calcula el tamaño de crop para las imágenes de baja resolución y el tamaño de imagen
    lr_crop_size = hr_crop_size // scale
    lr_image_shape = tf.shape(lr_img)[:2]

    # Se determina el ancho y alto de la imagen de baja resolución
    lr_width = tf.random.uniform(shape=(), maxval=lr_image_shape[0] - lr_crop_size + 1, dtype=tf.int32)
    lr_height = tf.random.uniform(shape=(), maxval=lr_image_shape[1] - lr_crop_size + 1, dtype=tf.int32)

    # Se calcula el ancho y alto de la imagen de alta resolución
    hr_width = lr_width * scale
    hr_height = lr_height * scale
    
    # Se toman los crops de las imágenes en alta y baja resolución
    lr_crop = lr_img[lr_width:lr_width + lr_crop_size, lr_height:lr_height + lr_crop_size]
    hr_crop = hr_img[hr_width:hr_width + hr_crop_size, hr_height:hr_height + hr_crop_size]

    return lr_crop, hr_crop


def random_flip(lr_img, hr_img):
    # Se calcula la probabilidad de giro
    flip_chance = tf.random.uniform(shape=(), maxval=1)  # Valores con decimales entre 0 y 1.
    
    return tf.cond(flip_chance < 0.5,
                   lambda: (lr_img, hr_img),  # Si probabilidad < 0.5, la imagen no se gira.
                   lambda: (tf.image.flip_left_right(lr_img),  # En caso contrario, se gira la imagen horizontalmente (de izq. a dcha.).
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    # Se determina el número de rotaciones de 90 grados (entre 0 y 4).
    rotate_option = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rotate_option), tf.image.rot90(hr_img, rotate_option)
