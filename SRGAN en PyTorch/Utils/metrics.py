from math import sqrt
from PIL import Image

def rmse_metric(predicted, actual):
    """
    Calcula el error cuadrático medio entre dos imágenes.

    Args:
        predicted (PIL.Image.Image): Imagen predicha.
        actual (PIL.Image.Image): Imagen real.

    Returns:
        float: Error cuadrático medio normalizado entre 0 y 1.
    """
    suma = 0.0

    predicted_pixels = predicted.load()
    actual_pixels = actual.load()

    # Calcula la suma de las diferencias al cuadrado entre los píxeles de las imágenes.
    for i in range(actual.size[0]):
        for j in range(actual.size[1]):
            suma += (predicted_pixels[i, j] - actual_pixels[i, j]) ** 2

    # Calcula el error cuadrático medio normalizado.
    mse = suma / (actual.size[0] * actual.size[1])
    rmse = sqrt(mse)
    return rmse / 255