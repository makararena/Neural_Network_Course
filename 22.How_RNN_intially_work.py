# Изначально надо понимать,зачем нам нужно RNN
# Представь словарь в котором 100 слов,а теперь 100 словосочетаний(100**2),а теперь 3 слова и таких 100 вариаций и тд,короче говоря оно не масштабируется
# Поэтому мы используем RNN,тк там веса и они просто меньше весят)

# Cуществуют разные RNN по строению( DT-RNN(один полносвязный(слой) в конце),DOT-RNN(несколько полносвязных в конце),Stacked RNN(слой за слоем))
# В этом видео расматривалась Stacked RNN - самая простая(просто связываем 2 слоя SimpleRNN),вот,как оно делается 
model = Sequential()
model.add(Embedding(maxWordsCount,256,input_length = inp_words))
model.add(SimpleRNN(128, activation='tanh',return_sequences = True)) # 'return_sequences = True' преобразует RNN из 'Many to One' в 'Many to many'
# чтобы связать с другим слоем
model.add(SimpleRNN(64, activation='tanh'))
model.add(Dense(maxWordsCount, activation='softmax'))
model.summary()
