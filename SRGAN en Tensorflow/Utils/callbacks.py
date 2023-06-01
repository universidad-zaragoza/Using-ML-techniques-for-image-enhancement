import tensorflow as tf

class SaveCustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager, steps_per_epoch):
        """
        Constructor de la clase SaveCustomCheckpoint.

        Args:
            checkpoint_manager: Objeto que gestiona los checkpoints.
            steps_per_epoch: Número de pasos por cada época.
        """
        self.checkpoint_manager = checkpoint_manager
        self.steps_per_epoch = steps_per_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Método llamado al final de cada época durante el entrenamiento del modelo.

        Args:
            epoch: Número de la época actual.
            logs: Diccionario con métricas de entrenamiento.
        """
        # Incrementa el número de época y el número total de pasos.
        self.checkpoint_manager.checkpoint.epoch.assign_add(1)
        self.checkpoint_manager.checkpoint.step.assign_add(self.steps_per_epoch)
        
        # Guarda el checkpoint actual.
        self.checkpoint_manager.save()
