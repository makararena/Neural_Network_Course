# Убираем ошибку о cpu
import os       
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras.layers import Dense # Создает слой нейронов в нейронной сети

# Обучающая выбока
c = np.array([-40, -10, 0, 8, 15, 22, 38]) #Цельсия 
f = np.array([-40, 14, 32, 46, 59, 72, 100])

model = keras.Sequential() # Создает модель послойной нейронной сети(самая обычная)
model.add(Dense(units=1, input_shape=(1,), activation='linear')) # units - сколько нейронов,input_shape - сколько входных нейронов,activation - активационная функция 
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1)) # loss - E(функция потерь),optimizer - оптимизатор по Адаму,0.1 - L(по умолчанию 0.001)
# Весовые коэфициенты - рандомные значения 

history = model.fit(c, f, epochs=500, verbose=0) # Эпохи - все данные(сколько раз пропускаем),verbose - не выводим служебную информацию
print("Обучение завершено")

print(model.predict([100])) # предиктим 100 градусов,чтобы получить фаренгуйты 
print(model.get_weights()) # Смотрим весовые коэфициенты для нашей сети

plt.plot(history.history['loss']) # Тут мы берем из dict history значения 'loss',которые у нас относяться к mse и вначале обращаемся к переменной history,в которой есть словарь history
plt.grid(True)
plt.show()
