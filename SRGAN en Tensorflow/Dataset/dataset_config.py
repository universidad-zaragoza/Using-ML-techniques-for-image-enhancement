""" 
Archivo de configuración del conjunto de datos.
"""

class DatasetConfig:
    def __init__(self, dataset_name, save_data_directory):
        if dataset_name not in datasets_disponibles.keys():
            raise ValueError(f"Los conjuntos de datos disponibles son: {datasets_disponibles.keys()}")

        dataset_parameters = datasets_disponibles[dataset_name]

        self.train_directory_lr = dataset_parameters["train_directory_lr"]  # Directorio de entrenamiento (baja resolución)
        self.valid_directory_lr = dataset_parameters["valid_directory_lr"]  # Directorio de validación (baja resolución)
        self.train_directory_hr = dataset_parameters["train_directory_hr"]  # Directorio de entrenamiento (alta resolución)
        self.valid_directory_hr = dataset_parameters["valid_directory_hr"]  # Directorio de validación (alta resolución)
        self.scale = dataset_parameters["scale"]  # Factor de escala de aumento

        self.save_data_directory = save_data_directory  # Directorio para guardar los datos

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
