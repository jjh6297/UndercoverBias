
from __future__ import print_function
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Input, LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, ZeroPadding2D, BatchNormalization, Activation, AveragePooling2D, Reshape, concatenate, DepthwiseConv2D, Concatenate
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow as tf
from tensorflow.keras.callbacks import History 
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
# from data import load_train_data, load_test_data
from itertools import combinations
from scipy.spatial import distance
# from theano import tensor as T
import sys
# from AsymDropoutLayer import Dropout_Asym_2D, Dropout_Asym_1D

sys.setrecursionlimit(10000)
batch_size = 512
nb_classes = 10
epochs = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB.
img_channels = 1

# The data, shuffled and split between train and test sets:

(X_train2, y_train2), (X_test2, y_test2) = fashion_mnist.load_data()
# X_test2 = X_test2.reshape(X_test2.shape[0], 784)
X_train2 = X_train2.astype('float32')
X_test2 = X_test2.astype('float32')
X_train2 /= 255.
X_test2 /= 255.
lll = 0.0
x_train_mean1 = np.mean(X_train2, axis=0)
X_train2 -= x_train_mean1
X_test2 -= x_train_mean1

(X_train3, y_train3), (X_test3, y_test3) = cifar10.load_data()
# X_test3 = X_test3.reshape(X_test2.shape[0], 784)
X_train3 = X_train3.astype('float32')
X_test3 = X_test3.astype('float32')
X_train3 /= 255.
X_test3 /= 255.
x_train_mean2 = np.mean(X_train3, axis=0)
X_train3 -= x_train_mean2
X_test3 -= x_train_mean2

X_train3 = X_train3.reshape(X_train3.shape[0], 32,32,3)

X_train4 = np.zeros((60000,32,32,3))
X_train4[:,2:-2,2:-2,0:1] = X_train2.reshape(X_train2.shape[0], 28,28,1)
X_train4[:,2:-2,2:-2,1:2] = X_train2.reshape(X_train2.shape[0], 28,28,1)
X_train4[:,2:-2,2:-2,2:3] = X_train2.reshape(X_train2.shape[0], 28,28,1)

X_train2=X_train4


X_test4 = np.zeros((10000,32,32,3))
X_test4[:,2:-2,2:-2,0:1] = X_test2.reshape(X_test2.shape[0], 28,28,1)
X_test4[:,2:-2,2:-2,1:2] = X_test2.reshape(X_test2.shape[0], 28,28,1)
X_test4[:,2:-2,2:-2,2:3] = X_test2.reshape(X_test2.shape[0], 28,28,1)

X_test2=X_test4

y_train2 = tensorflow.keras.utils.to_categorical(y_train2, 10)
y_test2 = tensorflow.keras.utils.to_categorical(y_test2, 10)
y_train3 = tensorflow.keras.utils.to_categorical(y_train3, 10)
y_test3 = tensorflow.keras.utils.to_categorical(y_test3, 10)

from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.regularizers import l1
trainableLabel = True
    
def Classifier():
    inputs = Input((32,32,3))

    dense1 = Conv2D(32,
                  kernel_size=3,
                  strides=1,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.),name='l1')(inputs)
    dense1 = SpatialDropout2D(0.2)(dense1)                  
    dense1 = Activation('relu')(dense1)
    dense1 = MaxPooling2D(2,2)(dense1)
    dense2 = Conv2D(64,
                  kernel_size=3,
                  strides=1,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.),name='l2')(dense1)
    dense2 = SpatialDropout2D(0.2)(dense2)
    dense2 = Activation('relu')(dense2)
    dense2 = Conv2D(64,
                  kernel_size=3,
                  strides=1,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.),name='l3')(dense2)
    dense2 = SpatialDropout2D(0.2)(dense2)
    dense2 = Activation('relu')(dense2)
    dense2 = MaxPooling2D(2,2)(dense2)

    dense3 = Conv2D(128,
                  kernel_size=3,
                  strides=1,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.),name='l4')(dense2)
    dense3 = SpatialDropout2D(0.2)(dense3)
    dense3 = Activation('relu')(dense3)
    dense3 = Conv2D(128,
                  kernel_size=3,
                  strides=1,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.),name='l5')(dense3)
    dense3 = SpatialDropout2D(0.2)(dense3)
    dense3 = Activation('relu')(dense3)
    dense3 = GlobalAveragePooling2D()(dense3)
    
    output= Dense(10,activation='softmax',name='l6')(dense3)
    model = Model([inputs], [output])

    return model
    

def AE():

    inputs1 = Input((32,32,3))
    inputs2 = Input((32,32,3))

    inputs = Concatenate()([inputs1, inputs2])

    dense1 = Conv2D(32,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE1')(inputs)
    dense1 = BatchNormalization()(dense1)                  
    dense1 = LeakyReLU()(dense1)
    dense2 = Conv2D(64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE2')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = LeakyReLU()(dense2)

    dense3 = Conv2D(128,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE3')(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = LeakyReLU()(dense3)

    dense4 = Conv2D(64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE4')(dense3)
    dense4 = BatchNormalization()(dense4)
    dense4 = LeakyReLU()(dense4)

    dense5 = Conv2D(32,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE5')(Concatenate()([dense2, dense4]))
    dense5 = BatchNormalization()(dense5)
    dense5 = LeakyReLU()(dense5)

    x1 = Conv2D(3,kernel_size=1,activation='sigmoid',
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE6')(Concatenate()([dense1, dense5]))


    dense6 = Conv2D(32,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE7')(x1)
    dense6 = BatchNormalization()(dense6)
    dense6 = LeakyReLU()(dense6)

    dense7 = Conv2D(64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE8')(dense6)
    dense7 = BatchNormalization()(dense7)
    dense7 = LeakyReLU()(dense7)

    dense8 = Conv2D(128,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE9')(dense7)
    dense8 = BatchNormalization()(dense8)
    dense8 = LeakyReLU()(dense8)
    
    
    dense9 = Conv2D(64,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE10')(dense8)
    dense9 = BatchNormalization()(dense9)
    dense9 = LeakyReLU()(dense9)

    dense10 = Conv2D(32,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE11')(Concatenate()([dense7, dense9]))    
    dense10 = BatchNormalization()(dense10)
    dense10 = LeakyReLU()(dense10)

    feature = Concatenate()([dense6, dense10])

    x2 = Conv2D(3,kernel_size=1,activation='sigmoid',
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE12')(feature)

    x3 = Conv2D(3,kernel_size=1,activation='sigmoid',
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE13')(feature)
                  
                  
    net1 = Classifier()
    net2 = Classifier()
    output1 = net1(x1)
    output1_2 = net1(x2)
    output1_3 = net1(inputs1)

    output2 = net2(Lambda(lambda x: x[0]-x[1])([x1,inputs1]))
    
    model = Model([inputs1,inputs2], [x1, x2, x3, output1, output1_2,output1_3, output2])
    return model
	
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop,SGD

from tensorflow.keras.callbacks import ModelCheckpoint


model = AE()
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss=['mae','mae','mae','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy','categorical_crossentropy'],loss_weights=[20., 1., 1., 0.03, 0.03, 0.03,5.],
			  optimizer=adam,metrics=['accuracy'])	

import random
								
from tensorflow.keras.preprocessing.image import ImageDataGenerator
								
# datagen = ImageDataGenerator(
        # # randomly rotate images in the range (deg 0 to 180)
        # rotation_range=30,
        # # randomly shift images horizontally
        # width_shift_range=0.4,
        # # randomly shift images vertically
        # height_shift_range=0.4,
        # # set range for random shear
        # shear_range=0.3,
        # # set range for random zoom
        # zoom_range=0.5,
        # horizontal_flip = True,
        # # set range for random channel shifts
        # # set mode for filling points outside the input boundaries
        # fill_mode='nearest',
        # # value used for fill_mode = "constant"
        # cval=0.,
# )
datagen = ImageDataGenerator(
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=15,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.2,
        # set range for random zoom
        zoom_range=0.3,
        horizontal_flip = True,
        # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
)

model.load_weights('pretrained.h5', by_name=True, skip_mismatch=True)

best = 9999.
for ii in range(100):

    X_train3_2, y_train3_2 = datagen.flow(X_train3,y_train3, batch_size=50000, shuffle=True)[0]
    X_train2_2, y_train2_2 = datagen.flow(X_train2,y_train2, batch_size=50000, shuffle=True)[0]
    
    # ranNeg = np.random.randint(2, size=(50000,1,1,1)).astype('float32')
    # X_train2_2 = np.abs(np.tile(ranNeg,[1,32,32,1])-X_train2_2)

    # print(len(X_train3_2))
    # X_train3_2 = X_train3_2.reshape(X_train2.shape[0], 784)
    # X_train2_2 = X_train2_2.reshape(X_train2.shape[0], 784)
    idx1 = list(range(X_train3_2.shape[0]))
    idx2 = list(range(X_train3_2.shape[0]))
    random.shuffle(idx1)
    random.shuffle(idx2)

    model.fit([X_train3_2[idx1,:,:,:],X_train2_2[idx2,:,:,:]], [X_train3_2[idx1,:,:,:],X_train3_2[idx1,:,:,:],X_train2_2[idx2,:,:,:],y_train3_2[idx1,:],y_train3_2[idx1,:],y_train3_2[idx1,:],y_train2_2[idx2,:]],
              batch_size=batch_size,
              epochs=1,
              verbose=1,shuffle=True)
              
    output = model.evaluate([X_test3,X_test2], [X_test3,X_test3,X_test2,y_test3,y_test3,y_test3,y_test2], batch_size=batch_size)
    print(output)
    
    if output[0]<best:
        best = output[0]
        model.save_weights('WatermarkingNet_Conv_CIFAR_best_New_V1_12.h5')
   
              
              
    model.save_weights('WatermarkingNet_CIFAR_Conv_New_V1_12.h5')
