# coding=utf-8
# dataset_config.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu and Pablo Doñate.
#
# Using Machine Learning techniques for image enhancement.
# This file has been created by jlaihong and modified by Pablo Doñate Navarro (800710@unizar.es).

"""
    Este archivo define una clase DatasetConfig que almacena los parámetros de los conjuntos de datos disponibles para la red SRGAN.
    También contiene un diccionario con los nombres y los parámetros de los conjuntos de datos disponibles.
"""

class DatasetConfig:
    def __init__(self, dataset_name, save_data_directory):
        # Comprueba si el nombre del conjunto de datos es válido
        if dataset_name not in datasets_disponibles.keys():
            raise ValueError(f"Los conjuntos de datos disponibles son: {datasets_disponibles.keys()}")

        # Obtiene los parámetros del conjunto de datos elegido
        dataset_parameters = datasets_disponibles[dataset_name]

        # Asigna los atributos de la clase con los parámetros del conjunto de datos
        self.train_directory_lr = dataset_parameters["train_directory_lr"]  # Directorio de entrenamiento (baja resolución)
        self.valid_directory_lr = dataset_parameters["valid_directory_lr"]  # Directorio de validación (baja resolución)
        self.train_directory_hr = dataset_parameters["train_directory_hr"]  # Directorio de entrenamiento (alta resolución)
        self.valid_directory_hr = dataset_parameters["valid_directory_hr"]  # Directorio de validación (alta resolución)
        self.scale = dataset_parameters["scale"]  # Factor de escala de aumento

        self.save_data_directory = save_data_directory  # Directorio para guardar los datos


# Define un diccionario con los nombres y los parámetros de los conjuntos de datos disponibles
datasets_disponibles = {
    "bicubic_x4": {
        "train_directory_lr": "DIV2K_train_LR",
        "valid_directory_lr": "DIV2K_valid_LR",
        "train_directory_hr": "DIV2K_train_HR",
        "valid_directory_hr": "DIV2K_valid_HR",
        "scale": 4
    },
    "chest_x-ray": {
        "train_directory_lr": "Chest_X-Ray_train_LR",
        "valid_directory_lr": "Chest_X-Ray_valid_LR",
        "train_directory_hr": "Chest_X-Ray_train_HR",
        "valid_directory_hr": "Chest_X-Ray_valid_HR",
        "scale": 4
    },
    "typical_x4": {
        "train_directory_lr": "Typical_train_LR",
        "valid_directory_lr": "Typical_valid_LR",
        "train_directory_hr": "Typical_train_HR",
        "valid_directory_hr": "Typical_valid_HR",
        "scale": 4
    }
}
