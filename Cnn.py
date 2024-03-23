import tensorflow as tf
from tf.keras import datasets, layers, models



def card_model():
  model = models.Sequential()
  #know input size, filter size came from poker cnn paper
  model.add(layers.Conv2D(filter = (3,3),activation = 'relu', input_shape = (17, 17, 6) kernel_size = (4,4)))
  model.add(layers.MaxPooling2D((2,2)))
  
  
  
  ''''
   model.add(layers.Conv2D(activation = 'relu', input_shape = (17, 17), kernel_size = (4,4)))


  #flatten before we use dense layers
  model.add(layers.flatten())
  '''
  model.add(layers.flatten())
  model.summary()
  return model

def game_model():


  




