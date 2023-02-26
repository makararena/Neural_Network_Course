import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Удаляем ошибку и использовании GPU

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten  #

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Загружаем данные 

# стандартизация входных данных( все делим на 255,тк 255 - максимальное значение теперь у нас все будет в промежутке от [0,1])
x_train = x_train / 255    
x_test = x_test / 255

# Переводим ответы в категориальные переменные( от 0 до 10 ---- [0,0,0,0,0,0,0,0,1,0])
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5)) # Создаем холст размером(10 : 5)
for i in range(25):
    plt.subplot(5,5,i+1) # создаем 25 холстов 
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary) # показываем наши 25 циферок 

plt.show()

model = keras.Sequential([              # Обрати внимание на слово 'sequential'
    Flatten(input_shape=(28, 28, 1)),   # Преобразует матрицу в слой,состоящий из векора 
    Dense(128, activation='relu'),      # Создаем слой(Dense),состоящий из 128 нейронов с функцией активации - ReLu
    Dense(10, activation='softmax')     # Создаем слой(Dense(связывает все нейроны между собой)) с функцией активации softmax(выводит 'вероятности')
])

print(model.summary())      # вывод структуры НС в консоль(сколько нейронных связей и где) 1.Dense - ((784 + 1) * 128), 2.Dense - 129 * 10 

model.compile(optimizer='adam',                 # Компилируем --- выстовляем оптимизатор по Адаму
            loss='categorical_crossentropy',    # E - категориальная кроссентропия
           metrics=['accuracy'])                # метрика - точность 

model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2,verbose=0)# Фитим,batch size - сколько изображений,validation_split - разбиение - 80% -train,20% - валидация(от 10 до 30)
model.evaluate(x_test, y_test_cat) # Тестим(проверка качества)

# Смотрим ответ ко второй фотографии
n = 1
x = np.expand_dims(x_test[n], axis=0) # создаем элемент тензора и добаляем еще одну ось,тк изначально у нас было 2 оси,тк predict method работает только с несколькими изображениями 
res = model.predict(x)
print(res)
print(np.argmax(res)) # Выводит индекс наибольшего значения в np.array

plt.imshow(x_test[n], cmap=plt.cm.binary) # Показываем изображение 
plt.show()


# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1) # Берем все индексы наибольших значений 

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# Выделение неверных вариантов
mask = pred == y_test # Сравниваем правильные и неправильные значение -> bool,которые показывают что True, а что False
print(mask[:10])

x_false = x_test[~mask] # Указываем значения,которые не правда 
y_false = x_test[~mask]

print(x_false.shape)

# Вывод первых 25 неверных результатов
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_false[i], cmap=plt.cm.binary) 

plt.show()



