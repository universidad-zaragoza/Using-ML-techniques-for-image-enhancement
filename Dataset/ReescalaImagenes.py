import os
from PIL import Image

def reescalaImagenesDirectorio(directorio, factorMultip):
    for i, filename in enumerate(os.listdir(os.getcwd() + directorio)):
        image = Image.open(str(os.getcwd() + directorio) + filename)
        width = image.size[0] // factorMultip
        height = image.size[1] // factorMultip
        image.thumbnail((width, height))    # Método que reescala la resolución de la imagen a la indicada.
        image.save(str(os.getcwd() + directorio) + filename, quality=100, optimize=True)
        
directorio_entrenamiento_lr = "/Chest_X-Ray/train_lr"
directorio_validación_lr = "/Chest_X-Ray/valid_lr"
directorio_test_lr = "/Chest_X-Ray/test_lr"

reescalaImagenesDirectorio(directorio_entrenamiento_lr, 4)
reescalaImagenesDirectorio(directorio_validación_lr, 4)
reescalaImagenesDirectorio(directorio_test_lr, 4)