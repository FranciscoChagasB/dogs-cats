import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

train_data = 'C:/Users/Francisco/Documents/catsAndDogs40/train'
test_data = 'C:/Users/Francisco/Documents/catsAndDogs40/test'

input_shape = (160, 160, 3)
batch_size = 24
epochs = 100

# Criando modelo.
model = Sequential()

# Convolução e Max-pooling.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Conexão das camadas.
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# A saída deve ser binária. Gato ou Cachorro.
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Aplicando pré-processamento nas imagens
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_data, target_size=(160, 160), batch_size=batch_size, class_mode='binary')

# Treinando o modelo
model.fit(train_generator, epochs=epochs)

# Avaliação do modelo
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_data, target_size=(160, 160), batch_size=batch_size, class_mode='binary')

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Acurácia no conjunto de teste: {test_accuracy}')

# Salve o modelo treinado
model.save('modelo_gato_cachorro.h5')