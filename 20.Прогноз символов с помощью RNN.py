import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

with open('train_data_true', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')  # убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '', text)  # заменяем все символы кроме кириллицы на пустые символы

# парсим текст, как последовательность символов
num_characters = 34  # 33 буквы + пробел
tokenizer = Tokenizer(num_words=num_characters,  # Возвращает n-ое кол-во элементов,если их будет больше,то остануться только наиболее повторяющиеся в тексте 
                      char_level=True)  # По чем будет разбивка(слова/символы) - слова
tokenizer.fit_on_texts([text])  # формируем токены на основе частотности в нашем тексте
print(tokenizer.word_index)     # В соответствии символу ставит число 

inp_chars = 6   # После скольки элеменов мы начинаем предиктить 
data = tokenizer.texts_to_matrix(text)  # преобразуем исходный текст в массив OHE
n = data.shape[0] - inp_chars  

X = np.array([data[i:i + inp_chars, :] for i in range(n)]) # То,что мы обработываем для предикта(4-5 символов)
Y = data[inp_chars:]  # предсказание следующего символа

print(data.shape)

# СОздаем модель 
model = Sequential()
model.add(Input((inp_chars,
                 num_characters)))  # при тренировке в рекуррентные модели keras подается сразу вся последовательность, поэтому в input теперь два числа. 1-длина последовательности, 2-размер OHE
model.add(SimpleRNN(128, activation='tanh'))  # рекуррентный слой на 128 нейронов
model.add(Dense(num_characters, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=100)


# Вспомогательная функция,которая будет строить фразу на основе значений 
def buildPhrase(inp_str, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j]))  # преобразуем символы в One-Hot-encoding

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters)

        pred = model.predict(inp)  # предсказываем OHE четвертого символа
        d = tokenizer.index_word[pred.argmax(axis=1)[0]]  # получаем ответ в символьном представлении

        inp_str += d  # дописываем строку

    return inp_str


res = buildPhrase("утренн")
print(res)




# Возвращает n-ое кол-во элементов,если их будет больше,то остануться только наиболее повторяющиеся в тексте 
# Исключает из текста такие символы 
# Как меду собой разделяются слова 

# Смотрим,какие индексы присвоил токенизатор
