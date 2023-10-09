# C чем вообще работает ML специалист в нейросети 
# 1. Со структурой HC 
# 2. Способами оптимизации алгоритма градиентного спуска 
# 3. Критерием качества 
# 4. Способом формирования выбрки валидации и обучающего множества 

# Как установить оптимизатор по Адаму правильно 
myAdam = keras.optimizers.Adam(learning_rate = 0.1) # Шаг сходимости(L) - learning_rate

model.compile(optimizer = myAdam,loss = 'categorical_crossentropy'
                metrics = ['accurancy'])

# Рекомендуют сначала использовать Adam, потом SGD(с моментами по Нестерову)
# После этого можно использовать все остальные : RMSProp,Adadelta,Adagrad,Adamax,Nadam,Ftrl


# Использование SGD по Нестерову 
myOpt = keras.optimizers.SGD(learning_rate = 0.1,momentum = 0.0,nesterov = True)


# Как лучше обозначать выборку валидации?
# 1.Самый простой и самый тупой способ - validation_split(самый тупой,потому что мы никоим образом не сможем увидеть значения валидации)
# 2.Выбирать вручную(тоже такое себе,тк 1. Делается вручную,2.Нету рандомизации,что очень важно)
# 3. train_test_split(1.Оно делается с помощью функции,2.Есть рандомизация)

# 2.
seq = 10000 # Размер выбоки валидации

x_train_split  = x_train[seq:]
y_train_split  = y_train[seq:]

x_train_val = x_train[:seq]
y_train_val = y_train[:seq]

# 3. 
x_train_split , x_train_val , y_train_split , y_train_val = train_test_split(X_train,y_train,test_size = 0.2)
