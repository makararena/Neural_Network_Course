# Создаем свою CNN,которая предугадывает цифры 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Отключение ошибки 

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D # 2D, тк у нас изображения(3D - видео,1D - аудио)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, axis=3) # Добавляем число каналов(1 для того,чтобы сеть работала)
x_test = np.expand_dims(x_test, axis=3)

print( x_train.shape )

model = keras.Sequential([
    # 32 - число каналов(ядер),(3,3) - размер ядра,
    # strides - шаг сканировния(по умолчанию 1 пиксель),
    # padding - добавляем маску(граничные элементы,чтобы все пиксели изображения были захвачены)
    # input_shape - входные значения 
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),  
    # pool_size = (2,2) - Размер нашего окна(если было изображение (128,128),то станет (64,64))
    # strides - None  - Шаг сканирования 
    # padding = 'valid' - нужно ли добавлять маску(нет)
    # data_format = None  - если есть какой-то необычный формат данных(batch,channels,rows,cols - 'channels first' либо batch,rows,cols,channels - 'channels_last'),по умолчанию второе
    MaxPooling2D((2, 2), strides=2),
    Conv2D(64, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2), # По итогу получиться тензор 7x7x64
    # Из всего тензора делает один вектор
    Flatten(),  
    # Создаем обычную HC
    Dense(128, activation='relu'),
    Dense(10,  activation='softmax')
])

# print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


his = model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

model.evaluate(x_test, y_test_cat)
