# The code is written based on the reference below 
# https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
# https://gist.github.com/virafpatrawala from Viraf Patrawala

from pycocotools.coco import COCO
import matplotlib.gridspec as gridspec
import numpy as np 
import tensorflow as tf
import os, sys
from tensorflow.keras import Input, Model
from keras.utils import np_utils
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import scipy.misc
import matplotlib.pyplot as plt
from scipy import misc
from matplotlib import pyplot
import scipy.ndimage
from tensorflow.keras.optimizers import Adam
import random
from keras.utils.np_utils import to_categorical  
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from mobilenetv2 import unet_model 

input_image_size=(224,224)
IMAGE_ORDERING = 'channels_last'
# given classes
#classes = ['laptop', 'dog', 'person', 'tv', 'cat' , 'car', 'bicycle']
classes = [ 'person']


n_class = len(classes)+1

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, input_image_size):
    # Read Image
    raw_img = plt.imread(imageObj)

    # convert to RGB instead of default BGR in opencv
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # Resize to image_size
    raw_img = cv2.resize(raw_img, input_image_size)

    # normalized to (-1,1) 
    normalized_img = tf.keras.applications.mobilenet_v2.preprocess_input(raw_img)

    if (len(normalized_img.shape)==3 and normalized_img.shape[2]==3): 
        return normalized_img, raw_img
    else: # To handle gray scale image, duplicate r,g, b, to the same 
        stacked_img = np.stack((normalized_img,)*3, axis=-1)
        return stacked_img, raw_img


model = unet_model(input_image_size[0], input_image_size[1], n_class)
model.load_weights('./model/pretrained.h5')
model.summary()

dataDir='/home/cvip/deep_learning/datasets/coco'
dataType='test2017'

imgPath= '{}/images/{}/'.format(dataDir,dataType)

images = []
dirlist = os.listdir(imgPath)

for diritem in dirlist:
    if diritem.endswith(".jpg"):
        images.append( imgPath + diritem )

# test this first sixty-four images
total_num_images = 64
plt.subplots(tight_layout=True)
for index in range(64):
    # inference only
    normalized_img, raw_img = getImage(images[index], input_image_size)

    x=np.reshape(normalized_img, (1,224,224,3))
    label = model.predict(x)
    c = index % 16
    r = index // 16
    plt.subplot(8,16,r * 32 + c+1)
    plt.axis('off')
    plt.imshow(raw_img)
    plt.subplot(8,16,r* 32 + c+16+1)
    plt.axis('off')
    plt.gray()
    plt.imshow(label[0,:,:,1])

plt.show()

