# Uso de técnicas de aprendizaje automático para la mejora de imágenes.

¡Bienvenido al repositorio "Using Machine Learning techniques for image enhancement"!

Este repositorio contiene una implementación de técnicas de aprendizaje automático para la mejora de imágenes. Está organizado en los siguientes directorios:

## [Dataset](./Dataset) :file_folder:
El directorio "Dataset" contiene conjuntos de datos utilizados para el entrenamiento, validación y pruebas del modelo de mejora de imágenes. Internamente, se encuentran los siguientes directorios:

- [**Entrenamiento de alta resolución**](./Dataset/Chest_X-Ray_train_HR): Contiene imágenes de alta resolución utilizadas para el entrenamiento del modelo.
- [**Entrenamiento de baja resolución**](./Dataset/Chest_X-Ray_train_LR): Contiene las versiones de baja resolución correspondientes a las imágenes de entrenamiento de alta resolución.
- [**Validación de alta resolución**](./Dataset/Chest_X-Ray_valid_HR): Contiene imágenes de alta resolución utilizadas para validar el rendimiento del modelo.
- [**Validación de baja resolución**](./Dataset/Chest_X-Ray_valid_LR): Contiene las versiones de baja resolución correspondientes a las imágenes de validación de alta resolución.
- [**Prueba de alta resolución**](./Dataset/Chest_X-Ray_test_HR): Contiene imágenes de alta resolución utilizadas para evaluar el modelo después del entrenamiento.
- [**Prueba de baja resolución**](./Dataset/Chest_X-Ray_test_LR): Contiene las versiones de baja resolución correspondientes a las imágenes de prueba de alta resolución.

Además, en el directorio "Dataset" encontrarás un script llamado "[ReescalaImagenes.py](./Dataset/ReescalaImagenes.py)", que se utiliza para reducir las dimensiones de una imagen.

## [SRGAN en Tensorflow](./SRGAN%20en%20Tensorflow) :gear:
El directorio "SRGAN en Tensorflow" contiene la implementación del modelo SRGAN (Super-Resolution Generative Adversarial Network) utilizando la biblioteca de TensorFlow. Puedes explorar este directorio para ver el código fuente y los archivos relacionados con la implementación de SRGAN en TensorFlow.
<br><br>El código del que se ha partido para su implementación ha sido: https://github.com/jlaihong/image-super-resolution.

## [SRGAN en PyTorch](./SRGAN%20en%20PyTorch) :gear:
El directorio "SRGAN en PyTorch" contiene la implementación del modelo SRGAN utilizando la biblioteca de PyTorch. Puedes explorar este directorio para ver el código fuente y los archivos relacionados con la implementación de SRGAN en PyTorch.
<br><br>En este caso, el código del que se ha partido para su implementación ha sido: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/srgan.

## [Resultados](./Resultados) :bar_chart:
El directorio "Resultados" almacena los resultados obtenidos después de aplicar las técnicas de mejora de imágenes. Puedes explorar este directorio para ver los resultados generados y comparar las imágenes mejoradas con las originales.

## 
¡Disfruta explorando este repositorio y experimentando con las técnicas de mejora de imágenes basadas en aprendizaje automático!
