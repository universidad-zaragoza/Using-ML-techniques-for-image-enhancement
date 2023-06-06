# coding=utf-8
# metrics.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu and Pablo Doñate.
#
# Using Machine Learning techniques for image enhancement.
# This file has been created by Pablo Doñate Navarro (800710@unizar.es).

"""
    Fichero que define las métricas de validación. 
    En este caso, se define el error cuadrático medio entre dos imágenes.
"""

from math import sqrt

def rmse_metric(predicted, actual):
    suma = 0.0

    predicted_pixels = predicted.load()
    actual_pixels = actual.load()
    
    for i in range(actual.size[0]):
        for j in range(actual.size[1]):
            suma += (predicted_pixels[i, j] - actual_pixels[i, j]) ** 2

    mse = suma / (actual.size[0] * actual.size[1])
    rmse = sqrt(mse)
    return rmse / 255