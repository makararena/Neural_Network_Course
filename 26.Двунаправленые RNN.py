# Двунаправленные нейронные сети нужны для,того,чтобы предиктить то,что посередине 
# Представь,что у тебя есть старые значения и будующие значние и тебе надо узнать,какое слово должно быт между ними 
# Архитектура простая,показана на фото

# Наблюдения подаются в формате диагональной матрицы(HC проще понять,какое наблюдение прошлое,а какое будущее(лучше понимает порядок))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, GRU, Input, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Создаем синусоиду с шумом 
N = 10000
data = np.array([np.sin(x/20) for x in range(N)]) + 0.1*np.random.randn(N)
plt.plot(data[:100])

# Формируем обучающую выборку 
off = 3     # Три осчета до и после
length = off*2+1    # Всего отчетов 
X = np.array([ np.diag(np.hstack((data[i:i+off], data[i+off+1:i+length]))) for i in range(N-length)])   # Входные значения(создаем диагональную матрицу)
Y = data[off:N-off-1]                                                                                   # Выходные значения 
print(X.shape, Y.shape, sep='\n')


model = Sequential()
model.add(Input((length-1, length-1)))
model.add(Bidirectional(GRU(2)) )               # Создаем двунаправленный слой
model.add(Dense(1, activation='linear'))        # Получаем 1-о входное значение 
model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(0.01))  # Компилируем


history = model.fit(X, Y, batch_size=32, epochs=10) # Фитим 

# Прогнозируем
M = 200
XX = np.zeros(M)
XX[:off] = data[:off]
for i in range(M-off-1):
  x = np.diag( np.hstack( (XX[i:i+off], data[i+off+1:i+length])) )  # Входные данные для HC 
  x = np.expand_dims(x, axis=0)                                     # Добавляем batch 
  y = model.predict(x)
  XX[i+off] = y

plt.plot(XX[:M])
plt.plot(data[:M])
plt.show()
