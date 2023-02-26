import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

# Include_top - использовать ли полносвязные слои(HC)(True,False)
# weights = 'imagenet' - используються уже обученные веса
# classes - число выходных нейронов 
# classifier_activation = 'softmax' - функция активации
#
model = keras.applications.VGG16()

im = Image.open(r'C:\red_wine.jpg')
im.show()


# приводим к входному формату VGG-сети
img = np.array(im)
x = keras.applications.vgg16.preprocess_input(img) # Мы из RGB создаем BGR + смещение(нужно для HC(которая VGG))
print(x.shape)
x = np.expand_dims(x, axis=0) # добавляем batch 
print(x.shape)

# прогоняем через сеть
res = model.predict(x)
print(np.argmax(res))

# По ссылке к уроку можно понять,какое изображение 

