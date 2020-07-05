#The code is written based on the  reference below
# https://www.tensorflow.org/tutorials/images/segmentation 
import numpy as np 
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, ReLU, Conv2DTranspose, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, AveragePooling2D, UpSampling2D, Add, BatchNormalization
from keras.utils import np_utils
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K


# the function below is the simpler version of 
# https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py
def upsample(filters, size):

  output = Sequential()
  output.add( Conv2DTranspose(filters, size, strides=2,
                            padding='same',
                            kernel_initializer=tf.random_normal_initializer(0., 0.02),
                            use_bias=False))

  output.add(BatchNormalization())
  output.add(ReLU())
  return output



def unet_model(input_height,input_width, n_class):
  feature_model = tf.keras.applications.MobileNetV2(input_shape=(input_height, input_width, 3), include_top=False, weights="imagenet")

  feature_model.summary()


  layer_names = [
      'block_1_expand_relu',   # 112x112
      'block_3_expand_relu',   # 56x56
      'block_6_expand_relu',   # 28x28
      'block_13_expand_relu',  # 14x14
      'block_16_project',     # 7x7
  ]

  layers = [feature_model.get_layer(name).output for name in layer_names]

  print(layers)

  # Create the feature extraction model
  down_stack = tf.keras.Model(inputs=feature_model.input, outputs=layers)
  # not trained 
  down_stack.trainable = False

  up_stack = [
      upsample(512, 3),  # 7x7 -> 14x14
      upsample(256, 3),  # 14x14 -> 28x28
      upsample(128, 3),  # 28x28 -> 56x56
      upsample(64, 3),   # 56x56 -> 112x112
  ]
  
  # start build the model. 

  inputs = tf.keras.layers.Input(shape=[input_height, input_width, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)

  print(skips)



  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model, the activation needs to be 'softmax' 
  last = tf.keras.layers.Conv2DTranspose(
      n_class, 3, strides=2,
      padding='same', activation='softmax') 

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
