import torch
from torchvision import transforms
from PIL import Image
from Modelos.generator import Generator

# Cargar modelos entrenados
generator_path = 'C:/Users/pablo/OneDrive - unizar.es/Trabajo_Doñate/TFG_2023/Propuesta/SRGAN_PYTORCH/saved_models/generator.pth'
discriminator_path = 'C:/Users/pablo/OneDrive - unizar.es/Trabajo_Doñate/TFG_2023/Propuesta/SRGAN_PYTORCH/saved_models/discriminator.pth'

# Cargar imagen
image_path = 'C:/Users/pablo/OneDrive - unizar.es/Trabajo_Doñate/TFG_2023/Propuesta/SRGAN_PYTORCH/data/Chest_X-Ray_test_HR/1.jpeg'
image = Image.open(image_path)
# Guardar la imagen de baja resolución transformada
hr_image_path = 'C:/Users/pablo/OneDrive - unizar.es/Trabajo_Doñate/TFG_2023/Propuesta/SRGAN_PYTORCH/images/test/hr_image.jpeg'
image.save(hr_image_path)

original_width, original_height = image.size

transform = transforms.Compose([
    transforms.Resize((original_height  // 4, original_width // 4)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Aplicar transformaciones a la imagen
image_transformed = transform(image)
# Guardar la imagen de baja resolución transformada
lr_transformed_image_path = 'C:/Users/pablo/OneDrive - unizar.es/Trabajo_Doñate/TFG_2023/Propuesta/SRGAN_PYTORCH/images/test/lr_image.jpeg'
transforms.ToPILImage()(image_transformed.squeeze()).save(lr_transformed_image_path)

# Cargar el generador y establecerlo en modo de evaluación
generador = Generator()
generador.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))

# Pasar la imagen transformada al generador
output_image = generador(image_transformed.unsqueeze(0))

# Guardar la imagen generada
gen_hr = output_image.squeeze()
gen_hr = (gen_hr - gen_hr.min()) / (gen_hr.max() - gen_hr.min())    # Escala de 0 a 1
gen_hr = (gen_hr * 255).clamp(0, 255).to(torch.uint8)               # Escala de 0 a 255

imagen_gen_hr = Image.fromarray(gen_hr.cpu().numpy(), mode='L')
imagen_gen_hr.save("C:/Users/pablo/OneDrive - unizar.es/Trabajo_Doñate/TFG_2023/Propuesta/SRGAN_PYTORCH/images/test/sr_image.jpeg")