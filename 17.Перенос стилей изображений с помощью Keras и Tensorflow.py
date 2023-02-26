import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras


#upl = files.upload()
#img = Image.open(BytesIO(upl['img.jpg']))
#img_style = Image.open(BytesIO(upl['img_style.jpg']))

img = Image.open(r'C:\jjj.jpg')
img_style = Image.open(r'C:\xxx.jpg')

# Показываем фотографии
plt.subplot(1, 2, 1)
plt.imshow( img )
plt.subplot(1, 2, 2)
plt.imshow( img_style )
plt.show()

# Конвертируем из RGB в BGR,добавляем смещения --- (preprocess_input) + добавляем нулевую ось(batch) --- np.expand_dims
x_img = keras.applications.vgg19.preprocess_input( np.expand_dims(img, axis=0) )
x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(img_style, axis=0))

# Вспомогательная функция,которая делает обратные шаги(из BGR в RGB и тд),чтобы мы могли потом увидеть фотографию
def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)    # Убираем нулевую ось 
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]") #
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # Добавляем обратно смещение + инверсим комноненты(из BGR в RGB)
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')    # Удаляем все,что меньше 0 и больше 255
  return x

# Загружаем эту HC,
vgg = keras.applications.vgg19.VGG19(include_top=False,     # Не используем обычную HC на конце 
                                     weights='imagenet')    # Оставляем те веса,на которох HC уже обучилась 
vgg.trainable = False                                       # 'веса,которые загружены,нельзя менять'

# Берем те самые фильтры,которые учавствуют в метриках(тут первая)
content_layers = ['block5_conv2'] 

# Тут вторая
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]
# Определяем количество этих слоев 
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Выделяем выходы из сети VGG-19
style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
model_outputs = style_outputs + content_outputs     # Суммируем их всех

# Выводим все выходы на экран 
print(vgg.input)
for m in model_outputs:
  print(m)

# Создаем свою модель из входа(vgg.input) и выходов,которые мы создали(model_outputs)
model = keras.models.Model(vgg.input, model_outputs)
for layer in model.layers:
    layer.trainable = False

print(model.summary())      # вывод структуры НС в консоль

def get_feature_representations(model):
  # batch compute content and style features
  style_outputs = model(x_style)
  content_outputs = model(x_img)
  
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features

# Вычисляем переменную Jc 
def get_content_loss(base_content, target):
# Сначала вычисляем сумму квадратов разности и потом делим на среднее арифметическое 
  return tf.reduce_mean(tf.square(base_content - target))

# Тут мы уже создаем матрицу Грамма 
def gram_matrix(input_tensor):
  channels = int(input_tensor.shape[-1])        # Берем количество каналов(в тензоре это у нас последняя колонка)
  a = tf.reshape(input_tensor, [-1, channels])  # Получаем ту матрицу,которая нам нужна,тк это не numpy,то первый аргумент - наш тензор,второй аргумент-
                                                # - shape,тк tf из одной матрицы делает другую,не оставляя никаких значений,то мы можем просто указать
                                                # последний элемент,как y(и автоматом добавиться второй),а ось x -  уже как каналы и в итоге получаем G

  n = tf.shape(a)[0]                            # Тут мы берем первую размерность от матрицы G,то есть y                            
  gram = tf.matmul(a, a, transpose_a=True)      # Тут у нас уже произведение матрицу на такую же,только транспонированную
  return gram / tf.cast(n, tf.float32)          # Тут мы уже Матрицу Грамма делим на значения y,тем самым нормализуем значения 

# Вычисляет 2-ю метрику для КАЖДОГО ОТДЕЛЬНОГО стиля
def get_style_loss(base_style, gram_target):
  gram_style = gram_matrix(base_style)          # Вычисляеться матрица грамма для формируемого изображения 
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))    # Вычисляем метрику по аналогии


# Тут уже формируем функцию,которая вычисляет общую метрику 
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  style_weight, content_weight = loss_weights # Содержит аргументы a,b
  
  model_outputs = model(init_image)           # Пропускает наше изображение через HC  
  
  style_output_features = model_outputs[:num_style_layers]      # Выделяем карту признаков для стилей 
  content_output_features = model_outputs[num_style_layers:]    # Выделяем карту признаков для контента 
  
  style_score = 0                                       # Создаем переменные,которые будут хранить потерю стилей
  content_score = 0                                     # Создаем переменные,которые будут хранить потерю контента

  # Вычисляем переменную Js
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Вычисляем переменную Jc 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
  # Перемножаем переменные a* Jc и b*Js
  style_score *= style_weight
  content_score *= content_weight

  # И по итогу складываем их
  loss = style_score + content_score 
  return loss, style_score, content_score

# Определяем вспомогательные переменные 'число итераций',веса для формулы 
num_iterations=100
content_weight=1e3
style_weight=1e-2

# Приводим код в действие передавай нашу функцию и модель
style_features, content_features = get_feature_representations(model)
# Вычисляем матрицу грамма для изображения со стилями 
gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

# Создаем копию нашего изображения 
init_image = np.copy(x_img)
# Инициализируем это изображение для tf 
init_image = tf.Variable(init_image, dtype=tf.float32)

# Создаем оптимизатор по Адаму 
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
iter_count = 1
# Определяем лучшие потери и фото,соответствующее лучшим потерям 
best_loss, best_img = float('inf'), None
loss_weights = (style_weight, content_weight)
# Заносим все в словарь
cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
}

norm_means = np.array([103.939, 116.779, 123.68])
min_vals = -norm_means
max_vals = 255 - norm_means
imgs = []

# Запускаем весь алгоритм градиентного спуска
for i in range(num_iterations):
    with tf.GradientTape() as tape: 
       all_loss = compute_loss(**cfg)
    
    loss, style_score, content_score = all_loss
    grads = tape.gradient(loss, init_image)

    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

      # Use the .numpy() method to get the concrete numpy array
      plot_img = deprocess_img(init_image.numpy())
      imgs.append(plot_img)
      print('Iteration: {}'.format(i))

plt.imshow(best_img)
print(best_loss)
plt.show()

image = Image.fromarray(best_img.astype('uint8'), 'RGB')
image.save("result.jpg")
