import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregue o modelo treinado
model = tf.keras.models.load_model('modelo_gato_cachorro.h5')

# Carregue uma imagem que você deseja classificar
image_path = 'C:/Users/Francisco/Documents/catsAndDogs40/test/dog/4.jpg'  # Substitua pelo caminho da sua imagem
img = image.load_img(image_path, target_size=(160, 160))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)  # Adicione uma dimensão para criar um lote de 1 imagem

# Faça a previsão
predictions = model.predict(img)

# As previsões são probabilidades, então você pode interpretá-las
if predictions[0][0] > 0.5:
    print('A imagem é um cachorro!')
else:
    print('A imagem é um gato!')