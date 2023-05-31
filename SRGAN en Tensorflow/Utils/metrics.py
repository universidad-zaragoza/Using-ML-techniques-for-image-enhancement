from math import sqrt
from PIL import Image

""" Determina el error cuadrático medio entre dos imágenes. """
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
