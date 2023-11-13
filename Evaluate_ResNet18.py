from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import fashion_mnist, mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow as tf
from tensorflow.keras.callbacks import History
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import h5py
import random
from tensorflow.keras.layers import Input, Dense, Permute, Reshape, ZeroPadding2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate, LeakyReLU, Input, Activation, Conv2D, Dense, Dropout, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import copy

import scipy.io as sio
nb_classes = 10
epochs = 200


import scipy.io as sio
import copy



ftr = sio.loadmat('Clean_CIFAR10.mat')
x_train_C = np.reshape(np.array(ftr['output']),[50000,32,32,3]).astype('float32')
y_train_C = np.array(ftr['label'])

ftr = sio.loadmat('Clean_CIFAR10_testSet.mat')
x_test_C = np.reshape(np.array(ftr['output']),[10000,32,32,3]).astype('float32')
y_test_C = np.array(ftr['label'])

ftr = sio.loadmat('Watermarked_CIFAR10.mat')
x_train_W = np.reshape(np.array(ftr['output']),[50000,32,32,3]).astype('float32')
y_train_W = np.array(ftr['label'])
x_train2 = np.reshape(np.array(ftr['watermark']),[np.array(ftr['watermark']).shape[0],32,32,3]).astype('float32')
y_train2 = np.array(ftr['WLabel'][0])


ftr = sio.loadmat('Watermarked_CIFAR10_testSet.mat')
x_test_W = np.reshape(np.array(ftr['output']),[10000,32,32,3]).astype('float32')
y_test_W = np.array(ftr['label'])
x_test2 = np.reshape(np.array(ftr['watermark']),[np.array(ftr['watermark']).shape[0],32,32,3]).astype('float32')
y_test2 = np.array(ftr['WLabel'][0])

Percent = 0.5
Idxs = random.sample(list(range(50000)), int(50000*Percent))
print(Idxs)
x_train = copy.deepcopy(x_train_C)
x_test = copy.deepcopy(x_test_C)
x_train_W2 = copy.deepcopy(x_train_W)
x_train[Idxs,:,:,:] = x_train_W2[Idxs,:,:,:]


x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

x_train_C -= x_train_mean
x_train_W -= x_train_mean
x_test_C-= x_train_mean
x_test_W -= x_train_mean








y_train = tensorflow.keras.utils.to_categorical(y_train_C, nb_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test_C, nb_classes)
y_train2 = tensorflow.keras.utils.to_categorical(y_train2, nb_classes)

y_test_W = tensorflow.keras.utils.to_categorical(y_test_W, nb_classes)
y_test2 = tensorflow.keras.utils.to_categorical(y_test2, nb_classes)

lll = 0.0
NumLength=5

trainableLabel = True
def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 330:
        lr *= 0.5e-3
    elif epoch > 500:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
from tensorflow.keras import regularizers



def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (3, 3),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):

    filters1, filters2 = filters
    bn_axis = 3
 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (3, 3), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',padding='same')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)

    shortcut = Conv2D(filters2, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1',padding='same')(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet18():

    img_input = Input(shape=(32,32,3))

    bn_axis = 3


    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7),
                      strides=(1, 1),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)

    x = identity_block(x, 3, [64, 64], stage=2, block='a')
    x = identity_block(x, 3, [64, 64], stage=2, block='b')

    x = conv_block(x, 3, [128, 128], stage=3, block='a', strides=(1, 1))
    x = identity_block(x, 3, [128, 128], stage=3, block='b')


    x = conv_block(x, 3, [256, 256], stage=4, block='a')
    x = identity_block(x, 3, [256, 256], stage=4, block='b')


    x = conv_block(x, 3, [512, 512], stage=5, block='a')
    x = identity_block(x, 3, [512, 512], stage=5, block='b')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(10, activation='softmax', name='fc1000')(x)
    model = Model(img_input, x, name='resnet18')


    return model

                
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip = True,
        shear_range=0.1,
        zoom_range=0.3,
        fill_mode='nearest',
        cval=0.,
)

import scipy.io as sio

DATA_NUM_CLASSES        = 10
DATA_CHANNELS           = 3
DATA_ROWS               = 32
DATA_COLS               = 32
DATA_CROP_ROWS          = 28
DATA_CROP_COLS          = 28
DATA_MEAN               = np.array([[[125.30691805, 122.95039414, 113.86538318]]]) # CIFAR10
DATA_STD_DEV            = np.array([[[ 62.99321928,  62.08870764,  66.70489964]]]) # CIFAR10    
MODEL_LEVEL_0_BLOCKS    = 4
MODEL_LEVEL_1_BLOCKS    = 6
MODEL_LEVEL_2_BLOCKS    = 3
TRAINING_BATCH_SIZE      = 32
TRAINING_SHUFFLE_BUFFER  = 5000
TRAINING_BN_MOMENTUM     = 0.99
TRAINING_BN_EPSILON      = 0.001
TRAINING_LR_MAX          = 0.001
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 10
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 100
TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE    
model = ResNet18()

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
lr_scheduler = LearningRateScheduler(lr_schedule)
batch_size = 512
callbacks = [lr_scheduler]
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])	
model.summary()




hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),steps_per_epoch=int(x_train.shape[0] / batch_size),epochs=100, verbose=1, workers=1, shuffle=True,validation_data = (x_test, y_test), callbacks=callbacks )


print('For Clean Training Dataset:')
output1 = model.evaluate(x_train_C,y_train)

print('For Watermarked Training Dataset:')
output2 = model.evaluate(x_train_W,y_train)

print('For Watermark Set(on Training Data):')
output3 = model.evaluate(x_train2,y_train2)

print('For Clean Test Dataset:')
output4 = model.evaluate(x_test_C,y_test)    

print('For Watermarked Test Dataset:')
output5 = model.evaluate(x_test_W,y_test)

print('For Watermark Set(on Test Data):')
output6 = model.evaluate(x_test2,y_test2)

sio.savemat('Evaluate_Results.mat', mdict = {'Acc_clean_tr':output1[1], 'CE_clean_tr':output1[0], 'Acc_watermarked_tr':output2[1], 'CE_watermarked_tr':output2[0],'Acc_watermark_tr':output3[1], 'CE_watermark_tr':output3[0],  'Acc_clean_te':output4[1], 'CE_clean_te':output4[0], 'Acc_watermarked_te':output5[1], 'CE_watermarked_te':output5[0],'Acc_watermark_te':output6[1], 'CE_watermark_te':output6[0],  'ValAcc':hist.history['val_accuracy'], 'Loss':hist.history['loss'], 'ValLoss':hist.history['val_loss'], 'Acc':hist.history['accuracy'] })
K.clear_session()