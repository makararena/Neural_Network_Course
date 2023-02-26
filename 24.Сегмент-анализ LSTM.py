import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загружаем текст с добрами высказываниями 
with open('train_data_true', 'r', encoding='utf-8') as f:
    texts_true = f.readlines()
    texts_true[0] = texts_true[0].replace('\ufeff', '') #убираем первый невидимый символ

# Загружаем текст со злыми высказываниями 
with open('train_data_false', 'r', encoding='utf-8') as f:
    texts_false = f.readlines()
    texts_false[0] = texts_false[0].replace('\ufeff', '') #убираем первый невидимый символ


texts = texts_true + texts_false                # Складываем тексты 
count_true = len(texts_true)                    # Определяем кол-во хороших фраз
count_false = len(texts_false)                  # Определяем кол-во плохих фраз
total_lines = count_true + count_false          # Определяем общее кол-во фраз
print(count_true, count_false, total_lines)

# Токенизируем все слова 
maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts) # Получаеться коллекция(tuple)

# Выводим частоту,просто,чтобы посмотреть,как работает 
dist = list(tokenizer.word_counts.items())
print(dist[:10])
print(texts[0][:100])


max_text_len = 10
data = tokenizer.texts_to_sequences(texts)                  # Переводит слова в наборы чисел
data_pad = pad_sequences(data, maxlen=max_text_len)         # Добавляет нули к нашей 'data'(нули ставит в начало)
print(data_pad)

print( list(tokenizer.word_index.items()) )


X = data_pad                                                # Входные значения(train_data)
Y = np.array([[1, 0]]*count_true + [[0, 1]]*count_false)    # Дублируем вектор([1,0]) count_true раз и дублируем вектор([0,1]) count_false раз,
                                                            #чтобы получить ответы True[1,0] или False[0,1]

print(X.shape, Y.shape)

# Перемешиваем наши наблюдения для работы нейронной сети
indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]


model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))   # Embeding - наши входные значения,
model.add(LSTM(128, return_sequences=True))     # units - кол-во нейронов в сигмоидной и тангенсоидной функции
                                                #'return_sequences = True' преобразует RNN из 'Many to One' в 'Many to many'  
model.add(LSTM(64))                             # Создаем 2-й слой LSTM
model.add(Dense(2, activation='softmax'))       # Создаем последний полносвязный слой
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))    # компилируем 

history = model.fit(X, Y, batch_size=32, epochs=50)         # фитим 

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

def sequence_to_text(list_of_indices):
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

t = "Я люблю позитивное настроение".lower()     # Переводим в нижний регистр 
data = tokenizer.texts_to_sequences([t])        # Пропускаем через токенайзер 
data_pad = pad_sequences(data, maxlen=max_text_len) # Преобразуем в нужный нам вектор 
print( sequence_to_text(data[0]) )              # Используем функцию,которая преоразует токенизированный текст обратно в текст 

res = model.predict(data_pad)                   # предиктим 
print(res, np.argmax(res), sep='\n')            # Смотрим ответ
