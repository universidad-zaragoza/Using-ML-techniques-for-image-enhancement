import os
import torch
from PIL import Image
from torchvision import transforms
from Modelos.generator import Generator

# Directorio actual del fichero test.py
current_directory = os.path.dirname(os.path.abspath(__file__))

# Cargar modelo de generador previamente entrenado
weights_directory = os.path.join(current_directory, "weights")
generador = Generator()
generator_path = os.path.join(weights_directory, "generator.pth")
generador.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))

# Cargar imagen
lr_image_path = os.path.join(current_directory, "TestImages/Chest_X-Ray_LR/1.jpeg")
lr_image = Image.open(lr_image_path)
# Guardar la imagen de baja resolución en el directorio de resultados (para realizar comparativas).
new_lr_image_path = os.path.join(current_directory, "TestImages/results/lr_image.jpeg")
lr_image.save(new_lr_image_path)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Aplicar transformaciones a la imagen
lr_image = transform(lr_image)

# Pasar la imagen transformada al generador
output_image = generador(lr_image.unsqueeze(0))

# Guardar la imagen generada
gen_hr = output_image.squeeze()
gen_hr = (gen_hr - gen_hr.min()) / (gen_hr.max() - gen_hr.min())    # Escala de 0 a 1
gen_hr = (gen_hr * 255).clamp(0, 255).to(torch.uint8)               # Escala de 0 a 255

imagen_gen_hr = Image.fromarray(gen_hr.cpu().numpy(), mode='L')
sr_path = os.path.join(current_directory, "TestImages/results/sr_image.jpeg")
imagen_gen_hr.save(sr_path)

hr_image_path = os.path.join(current_directory, "TestImages/Chest_X-Ray_HR/1.jpeg")
hr_image = Image.open(hr_image_path)
new_hr_image_path = os.path.join(current_directory, "TestImages/results/hr_image.jpeg")
hr_image.save(new_hr_image_path)