# coding=utf-8
# dataset_loader.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu and Pablo Doñate.
#
# Using Machine Learning techniques for image enhancement.
# This file has been created by jlaihong and modified by Pablo Doñate Navarro (800710@unizar.es).

"""
    Este archivo define funciones para crear y cargar datasets de 
    imágenes de baja y alta resolución para la red SRGAN.
"""

import os
import tensorflow as tf
import glob
from tensorflow.python.data.experimental import AUTOTUNE

def image_dataset_from_directory(data_directory, image_directory):
    # Obtiene la ruta de las imágenes
    images_path = os.path.join(data_directory, image_directory)

    # Obtiene los nombres de los archivos de imagen en la ruta
    filenames = sorted(glob.glob(images_path + "/*.jpeg"))

    # Crea un dataset a partir de los nombres de los archivos
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    # Lee los archivos de imagen
    dataset = dataset.map(tf.io.read_file)
    # Decodifica las imágenes en formato JPEG
    dataset = dataset.map(lambda x: tf.image.decode_jpeg(x, channels=1), num_parallel_calls=AUTOTUNE)

    # Directorio de caché para almacenar datos procesados y acelerar la carga
    cache_directory = os.path.join(data_directory, "cache", image_directory)
    os.makedirs(cache_directory, exist_ok=True)

    # Ruta completa del archivo de caché
    cache_file = cache_directory + "/cache"

    # Almacena en caché el dataset para acelerar la carga en ejecuciones posteriores
    dataset = dataset.cache(cache_file)

    # Si el archivo de caché no existe, popula la caché con los datos del dataset
    if not os.path.exists(cache_file + ".index"):
        populate_cache(dataset, cache_file)

    return dataset


def create_training_dataset(dataset_parameters, batch_size, train_mappings):
    # Crea un dataset de imágenes de baja resolución (LR) desde el directorio de entrenamiento
    lr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, dataset_parameters.train_directory_lr)
    # Crea un dataset de imágenes de alta resolución (HR) desde el directorio de entrenamiento
    hr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, dataset_parameters.train_directory_hr)

    # Combina los datasets de LR y HR en un solo dataset
    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    # Aplica las transformaciones de mapeo especificadas para el entrenamiento
    for mapping in train_mappings:
        dataset = dataset.map(mapping, num_parallel_calls=AUTOTUNE)

    # Agrupa los datos en lotes y repite el dataset para múltiples épocas
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def create_validation_dataset(dataset_parameters):
    # Crea un dataset de imágenes de baja resolución (LR) desde el directorio de validación
    lr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, dataset_parameters.valid_directory_lr)
    # Crea un dataset de imágenes de alta resolución (HR) desde el directorio de validación
    hr_dataset = image_dataset_from_directory(dataset_parameters.save_data_directory, dataset_parameters.valid_directory_hr)

    # Combina los datasets de LR y HR en un solo dataset
    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    # Agrupa los datos en lotes y repite el dataset una vez
    dataset = dataset.batch(1)
    dataset = dataset.repeat(1)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def create_training_and_validation_datasets(dataset_parameters, train_batch_size, train_mappings):
    # Crea el dataset de entrenamiento
    training_dataset = create_training_dataset(dataset_parameters, train_batch_size, train_mappings)
    # Crea el dataset de validación
    validation_dataset = create_validation_dataset(dataset_parameters)

    return training_dataset, validation_dataset


def populate_cache(dataset, cache_file):
    print(f'Comienza el almacenamiento en caché en {cache_file}.')
    # Recorre el dataset completo para almacenar los datos en la caché
    for _ in dataset:
        pass
    print(f'Almacenamiento en caché completado en {cache_file}.')
