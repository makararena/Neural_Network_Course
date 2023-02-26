# Тут почти все тоже самое,только мы вместо индексации букв переходи к индексации слов и тд
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input,Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical

with open('text', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

maxWordsCount = 1000                       # Вообще было бы неплохо 20 000 
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)                                                  # char_level = False,тк мы используем слова 
tokenizer.fit_on_texts([texts])

dist = list(tokenizer.word_counts.items())                # Получаеться лист туплов 
print(dist[:10])

data = tokenizer.texts_to_sequences([texts])              # Преобразуем текст в последовательность чисел 
#res = to_categorical(data[0], num_classes=maxWordsCount)  # Преобразуем в One Hot векторы [0,0,1,0,0,0.....до 1000]
#print(res.shape)
res = np.array(data[0])
inp_words = 3                                               # Берем количество слов 
n = res.shape[0] - inp_words


X = np.array([res[i:i + inp_words] for i in range(n)])          # Берем тензор входных слов   
Y = to_categorical(res[inp_words:],num_classes = maxWordsCount)     # Создаем тензор выходных слов

# Строим RNN
model = Sequential()
model.add(Embedding(maxWordsCount,256,input_length = inp_words))
#model.add(Input((inp_words, maxWordsCount)))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(maxWordsCount, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=50)

# Создаем функцию,которая все делает за нас 
def buildPhrase(texts, str_len=20):
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_len):
        #x = to_categorical(data[i: i + inp_words], num_classes=maxWordsCount)  # преобразуем в One-Hot-encoding
        #inp = x.reshape(1, inp_words, maxWordsCount)
        x = data[i: i + inp_words]
        inp = np.expand_dims(x,axis = 0)

        pred = model.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res += " " + tokenizer.index_word[indx]  # дописываем строку

    return res


res = buildPhrase("позитив добавляет годы")
print(res)



# Тензор,который мы сформировали с информацией(X,Y) при 20 000 словах уже занимает 100 мб,а если расширять,то будет совсем плохо
# Поэтому можно использовать Embedding,который просто подает какой-то индекс нейрона на вход,и тем самым мы не будем огромный вектор хранить у себя

keras.layers.Embedding(input_dim,output_dim,input_length)

# input_dim - число слов в словаре 
# output_dim - число выходов в полносвязном Embedding - слое 
# input_length - размер входного вектора(число слов,по которым строится прогноз )
