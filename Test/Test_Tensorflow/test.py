import os
import tensorflow as tf
from tensorflow.python.data.experimental import AUTOTUNE
from PIL import Image
from Modelos.generator import build_generator
from Dataset.dataset_loader import image_dataset_from_directory

# Directorio actual del fichero test.py
current_directory = os.path.dirname(os.path.abspath(__file__))

def create_test_dataset(data_directory, image_directory):
    lr_dataset = image_dataset_from_directory(data_directory, image_directory)
    hr_dataset = image_dataset_from_directory(data_directory, image_directory)

    dataset = tf.data.Dataset.zip((lr_dataset, hr_dataset))

    dataset = dataset.batch(1)
    return dataset


# Rutas de directorios
data_directory = os.path.join(current_directory, "TestImages")
image_directory = "Chest_X-Ray_LR"

dataset = create_test_dataset(data_directory, image_directory)
test_dataset = dataset.take(1)

# Rutas de directorios y archivos para modelos entrenados
weights_directory = os.path.join(current_directory, "weights")
generator = build_generator()
generator_path = os.path.join(weights_directory, "generador.h5")
generator.load_weights(generator_path)

for lr, hr in test_dataset: 
    # Procesar imagen con el generador
    output_image = generator(lr)

# Guardar imagen generada
sr = tf.clip_by_value(output_image, 0, 255)
sr = tf.round(sr)
sr = tf.cast(sr, tf.uint8)

image_sr = Image.fromarray(sr.numpy().squeeze())

sr_path = os.path.join(current_directory, "TestImages/results/sr_image.jpeg")
image_sr.save(sr_path)

# Guardar imagen baja resolucion
lr = tf.clip_by_value(lr, 0, 255)
lr = tf.round(lr)
lr = tf.cast(lr, tf.uint8)

image_lr = Image.fromarray(lr.numpy().squeeze())

lr_path = os.path.join(current_directory, "TestImages/results/lr_image.jpeg")
image_lr.save(lr_path)

hr_image_path = os.path.join(current_directory, "TestImages/Chest_X-Ray_HR/1.jpeg")
hr_image = Image.open(hr_image_path)
new_hr_image_path = os.path.join(current_directory, "TestImages/results/hr_image.jpeg")
hr_image.save(new_hr_image_path)
