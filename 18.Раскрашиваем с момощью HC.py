# Градации серого представлены одним слоем матрицы(0(черный)-255(белый)),а RGB - 3 цвета(на выходе HC должно быть 3 канала)
# Но использование цветового пространства Lab(light([0;100]),a,b([-128;127])) - намного удобнее,тк на выходе м получим только 2 слоя,Light - 
# черно-белый еще в себя включает яркость изображения 


# Почему раскраска вообще работает? HC в главном блоке(512 нейронов) сохраняет все характерные фигуры и связывает их с цветами(просто запоминает)


from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
#from google.colab import files
#from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Загружаем изображение 
upl = files.upload()
names = list(upl.keys())
img = Image.open(BytesIO(upl[names[0]]))

# Преобразуем в формат Lab 
def processed_image(img):
  image = img.resize( (256, 256), Image.BILINEAR)
  image = np.array(image, dtype=float)
  size = image.shape
  lab = rgb2lab(1.0/255*image)          # Переводим в Lab 
  X, Y = lab[:,:,0], lab[:,:,1:]        # Выделяет X - яркость,Y - цвета 

  Y /= 128                              # нормируем выходные значение в диапазон от -1 до 1
  X = X.reshape(1, size[0], size[1], 1) # Создаем размерности(batch,row,column,channels)
  Y = Y.reshape(1, size[0], size[1], 2)
  return X, Y, size

# Возвращаем нужные нам переменные 
X, Y, size = processed_image(img)

# Формируем HC(по итогу мы уменьшаем карту признаков 3 раза,а потом увеличиваем,по итогу получаем тоже самое )
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))                                  # На вход подаем изображение X,y
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))                    # Первый сверточный слой(64 фильтра),padding = 'same' - получаем 
# такое же количество пикселей,сколько у нас было до сканирования 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))         # Тоже самое,только теперь шаг сканирования - 2,но не Max Poling,
# тк Max Poling изменяет само изображение,а этого при colorisation нам нельзя допустить 
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))                                                      # Увеличивает карту признаков(количество пикселей в 2 раза)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))                      # Гиперболический тангенс - [0,1]
model.add(UpSampling2D((2, 2)))

model.compile(optimizer='adam', loss='mse')                                          # Компилируем 
model.fit(x=X, y=Y, batch_size=1, epochs=50)                                         # Фитим

#upl = files.upload()
#names = list(upl.keys())
#img = Image.open(BytesIO(upl[names[0]]))
X, Y, size = processed_image(img)

output = model.predict(X)            # Предиктим яркость,чтобы получить a,b

output *= 128                        # Возвращаем им нормальный диапазон
print(output)
min_vals, max_vals = -128, 127
ab = np.clip(output[0], min_vals, max_vals)     # обрезаем наши выходы если вдруг перескочат значения

cur = np.zeros((size[0], size[1], 3))           # Генерируем нулеве изображение 
cur[:,:,0] = np.clip(X[0][:,:,0], 0, 100)       # Загружаем яркость 
cur[:,:,1:] = ab                                # Загружаем цвета 
plt.subplot(1, 2, 1)                            # Плотим 
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(lab2rgb(cur))                        # Плотим новое изображение 
