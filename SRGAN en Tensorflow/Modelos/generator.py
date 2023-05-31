import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda
from tensorflow.python.keras.layers import PReLU

upsamples_per_scale = {
    2: 1,
    4: 2,
    8: 3
}

def residual_block(input):
    rb = Conv2D(filters=64, kernel_size=3, padding="same")(input)
    rb = BatchNormalization(momentum=0.8)(rb)
    rb = PReLU(shared_axes=[1, 2])(rb)
    rb = Conv2D(filters=64, kernel_size=3, padding="same")(rb)
    rb = BatchNormalization(momentum=0.8)(rb)
    return Add()([input, rb])

def upsample_block(input):
    ub = Conv2D(filters=256, kernel_size=3, padding='same')(input)
    ub = pixel_shuffler(scale=2)(ub)
    return PReLU(shared_axes=[1, 2])(ub)

def pixel_shuffler(scale):
    return Lambda(lambda x: tf.compat.v1.depth_to_space(x, scale))

def normaliza(x):
    return x / 255.0

def denormaliza(x):
    return (x + 1) * 127.5

def build_generator(scale=4, num_filters=64, num_residual_blocks=16):
    if scale not in upsamples_per_scale:
        raise ValueError(f"Los factores de multiplicación válidos son: {upsamples_per_scale.keys()}")

    num_upsample_blocks = upsamples_per_scale[scale]

    # Se define el tipo de entrada y se normalizan sus valores
    input_layer = Input(shape=(None, None, 1))
    x = Lambda(normaliza)(input_layer)

    # Capa convolucional con función de activación 'PReLU'.
    first_layer = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    first_layer = PReLU(shared_axes=[1, 2])(first_layer)

    # Primer bloque residual
    x = residual_block(first_layer)
    
    # Bucle con x bloques residuales
    for _ in range(num_residual_blocks - 1):
        x = residual_block(x)

    # Último bloque residual sin función de activación
    output_residual = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    output_residual = BatchNormalization()(output_residual)
    output_residual = Add()([output_residual, first_layer])

    # Capa de upsample
    upsample_layer = upsample_block(output_residual)

    # Bucle con x bloques de upsample
    for _ in range(num_upsample_blocks - 1):
        upsample_layer = upsample_block(upsample_layer)

    # Última capa de la red generadora con función de activación tanh.
    gen_output = Conv2D(1, kernel_size=9, padding='same', activation='tanh')(upsample_layer)

    # Se denormalizan los valores
    sr = Lambda(denormaliza)(gen_output)

    return Model(inputs=input_layer, outputs=sr)
