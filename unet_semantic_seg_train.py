# The code is written based on the reference below 
# https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
# https://gist.github.com/virafpatrawala from Viraf Patrawala

from pycocotools.coco import COCO
import matplotlib.gridspec as gridspec
import numpy as np 
import tensorflow as tf
from tensorflow.keras import Input, Model
from keras.utils import np_utils
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from matplotlib import pyplot

from tensorflow.keras.optimizers import Adam
import random
from keras.utils.np_utils import to_categorical  
import cv2


from mobilenetv2 import unet_model 

#current code only supports two classes problem : background and foreground, if you want to support more than two classes,
# the getMask function needs to be changed ( the way to do interpolation )

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

def getImage(imageObj, img_folder, input_image_size):
    # Read Image
    train_img = plt.imread(img_folder + '/' + imageObj['file_name'])

    # convert to RGB instead of default BGR in opencv
    train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
    # Resize to image_size
    train_img = cv2.resize(train_img, input_image_size)

    # normalized to (-1,1) 
    train_img = tf.keras.applications.mobilenet_v2.preprocess_input(train_img)
    

    if (len(train_img.shape)==3 and train_img.shape[2]==3): 
        return train_img
    else: # To handle gray scale image, duplicate r,g, b, to the same 
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img

def getMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)+1
        # 128
        new_mask = cv2.resize(coco.annToMask(anns[a])*128, input_image_size)
        # handle borderline case
        new_mask[np.where(new_mask<64)]=0 
        new_mask[np.where(new_mask>=64)]=pixel_value

        mask = np.maximum(new_mask, mask)

    train_mask = mask.reshape(input_image_size[0], input_image_size[1], 1)
    
    train_mask = to_categorical(train_mask, num_classes=n_class)
    #train_mask[:,:,0]=0
    return train_mask


def calSamplenum(images, classes, coco, batch_size=4, folder=''):
    image_size = len(images)
    catIds = coco.getCatIds(catNms=classes)

    c = 0
    num_background = 0
    num_foreground = 0

    while (True):

        for i in range(c, c + batch_size):
            imageObj = images[i]

            ### Generate Mask ###

            train_mask = getMask(imageObj, classes, coco, catIds, input_image_size)

            num_foreground += np.sum(train_mask[:,:,1])
            num_background += np.sum(train_mask[:, :, 0])


        c += batch_size
        if (c + batch_size >= image_size):
            break
    return num_background, num_foreground



def imageGeneratorCoco(images, classes, coco,batch_size=4, folder=''):
    
    image_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    
    c = 0
 
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], n_class)).astype('float')

        for i in range(c, c+batch_size): 
            imageObj = images[i]
            
            ### Extract Image ###
            train_img = getImage(imageObj, folder, input_image_size)
                
            ### Generate Mask ###

            train_mask = getMask(imageObj, classes, coco, catIds, input_image_size)                
            
            # Add to respective batch sized arrays
            img[i-c] = train_img
            mask[i-c] = train_mask
            
        c = c + batch_size
        if(c + batch_size >= image_size):
            c=0
            random.shuffle(images)

        yield img, mask

def visualizeGenerator(gen):
    img, mask = next(gen)
    
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
    
    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:,:,1])
                
            ax.axis('off')
            fig.add_subplot(ax)        
    plt.show()



model = unet_model(input_image_size[0], input_image_size[1], n_class)
model.load_weights('./model/pretrained.h5')
model.summary()

def create_image_generator(dataType):
    dataDir='/home/cvip/deep_learning/datasets/coco'

    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

    imgPath= '{}/images/{}/'.format(dataDir,dataType)
    coco=COCO(annFile)

    temp_images = []

    for className in classes:
        # get all images containing given class
        catIds = coco.getCatIds(catNms=className)
        imgIds = coco.getImgIds(catIds=catIds)

        temp_images += coco.loadImgs(imgIds)

    total_num_images = len(temp_images)

    images = []
    for i in range(total_num_images):
        if temp_images[i] not in images:
            images.append(temp_images[i])

    # image number
    image_num = len(images)

    print(image_num)

    return images, coco, imgPath

batch_size = 4

'''
calculate how many samples for each class, try to solve the imbalance of background and foreground classes, the sample
ratio is used inside w_categorical_crossentropy below : loss = w_categorical_crossentropy([1,5])

n_bg, n_fg = calSamplenum(images, classes, coco, batch_size, imgPath)
print(n_bg, n_fg, n_fg/n_bg)
'''
train_images, train_coco , train_imgPath = create_image_generator('train2017')
val_images, val_coco, val_imgPath = create_image_generator('val2017')

train_gen = imageGeneratorCoco(train_images, classes, train_coco, batch_size, train_imgPath)

val_gen = imageGeneratorCoco(val_images, classes, val_coco, batch_size, val_imgPath)

if 1 : 
    visualizeGenerator(train_gen)
    visualizeGenerator(val_gen)


train_image_num = len(train_images)
steps_per_epoch = train_image_num // batch_size

val_image_num = len(val_images)
validation_steps = val_image_num // batch_size
epochs = 512
learning_rate = 0.00001
decay_rate = learning_rate/ epochs
opt = Adam(lr= learning_rate, decay = decay_rate)


def accuracy_ignoring_zerolabel(y_true, y_pred):
# change to one dimen tensor    
    y_pred = K.reshape(y_pred, (-1, n_class))
    y_true = K.reshape(y_true, (-1, n_class))
  
# check the number of label with nonzero   
    nonzero_labels = K.sum(y_true, axis=1)
    nonzero_labels = K.greater(nonzero_labels, 0)

    matched_labels = K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1) )

    n_nonzero_labels = K.sum(tf.cast(nonzero_labels, tf.float32))

    matched_nonzero_labels = nonzero_labels & matched_labels

    n_matched_nonzero_labels = K.sum(tf.cast(matched_nonzero_labels, tf.float32) )

    return n_matched_nonzero_labels/ n_nonzero_labels

def w_categorical_crossentropy(weights):

        
    def loss(y_true, y_pred):
  
        # avoid log(0) and log(1)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon() )

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

model.compile(loss = w_categorical_crossentropy([1,5]), optimizer = opt, metrics= [accuracy_ignoring_zerolabel])


mc = ModelCheckpoint('wt{epoch:05d}.h5', save_weights_only=True, save_freq=steps_per_epoch)


history = model.fit(x = train_gen,
                    validation_data = val_gen,
                    steps_per_epoch = steps_per_epoch,
                    validation_steps = validation_steps,
                    callbacks = [mc],
                    epochs = epochs,
                    verbose = True)

