
from __future__ import print_function
import tensorflow
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Input, LeakyReLU, GlobalMaxPooling2D, GlobalAveragePooling2D, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, ZeroPadding2D, BatchNormalization, Activation, AveragePooling2D, Reshape, concatenate, DepthwiseConv2D, Concatenate
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import tensorflow as tf
from tensorflow.keras.callbacks import History 
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from itertools import combinations
from scipy.spatial import distance
import sys
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop,SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import scipy.io as sio
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.regularizers import l1
sys.setrecursionlimit(10000)
batch_size = 512
nb_classes = 10
epochs = 200
data_augmentation = False
PartialRate = 1.
img_rows, img_cols = 32, 32
img_channels = 1


(X_train2, y_train2), (X_test2, y_test2) = fashion_mnist.load_data()
X_train2 = X_train2.astype('float32')
X_test2 = X_test2.astype('float32')
X_train2 /= 255.
X_test2 /= 255.
lll = 0.0
x_train_mean1 = np.mean(X_train2, axis=0)
X_train2 -= x_train_mean1
X_test2 -= x_train_mean1

(X_train3, y_train3), (X_test3, y_test3) = cifar10.load_data()
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

II = []
II2 = []
for ii in range(10):
    idx1 = np.where(y_train2==ii)[0]
    idx2 = np.where(y_train3==ii)[0]   
    print(idx1)
    choiceLIst = [random.choice(list(idx1)) for i in range(len(idx2))]
    II = II+choiceLIst
    II2 = II2+list(idx2)
    
print(len(II))
X_train2 = X_train2[II,:,:,:]  
y_train2 = y_train2[II]  
X_train3 = X_train3[II2,:,:,:]  
y_train3 = y_train3[II2]  



II = []
II2 = []
for ii in range(10):
    idx1 = np.where(y_test2==ii)[0]
    idx2 = np.where(y_test3==ii)[0]   
    print(idx1)
    choiceLIst = [random.choice(list(idx1)) for i in range(len(idx2))]
    II = II+choiceLIst
    II2 = II2+list(idx2)
    
print(len(II))
X_test2 = X_test2[II,:,:,:]  
y_test2 = y_test2[II]  
X_test3 = X_test3[II2,:,:,:]  
y_test3 = y_test3[II2]  



    
    
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

    x1 = Conv2D(3,kernel_size=1,activation='tanh',
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

    x2 = Conv2D(3,kernel_size=1,activation='tanh',
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),name='AE12')(feature)

    x3 = Conv2D(3,kernel_size=1,activation='tanh',
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



# model = tensorflow.keras.models.load_model('PretrainedModel.h5')
model = AE()
adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss=['mae','mae','mae','categorical_crossentropy','mae','categorical_crossentropy','mae'],loss_weights=[8., 0.5, 0.2, 0.05, 0.05, 0.03, 0.03 ],
			  optimizer=adam,metrics=['accuracy'])	

								


model.load_weights('pretrained.h5')
output = model.predict([X_train3,X_train2],batch_size=batch_size)



II3 = random.sample(list(range(X_train3.shape[0])), int((1.-PartialRate)*float(X_train3.shape[0])))
output[0][II3,:,:,:] = X_train3[II3,:,:,:]
II4 = list(range(X_train3.shape[0]))

for ii in range(len(II3)):
	del II4[II4.index(II3[ii])]



output=output[0]

# output2 = output[II4,:,:,:]-X_train3[II4,:,:,:]
# y_train2 = y_train2[II4]

# Scale = 1.

# output[II4,:,:,:] = output[II4,:,:,:] + (Scale- 1.)*output2
# output2 = Scale*output2 




output=(output+x_train_mean2)*255
output.astype('uint8')
output=output/255.-x_train_mean2

output2 = output[II4,:,:,:]-X_train3[II4,:,:,:]
y_train2 = y_train2[II4]


print(len(II4))
sio.savemat('Watermarked_CIFAR10.mat', mdict = {'output':output, 'label':y_train3, 'watermark':output2, 'WLabel': y_train2})
sio.savemat('Clean_CIFAR10.mat', mdict = {'output':X_train3, 'label':y_train3, 'watermark':output2, 'WLabel': y_train2})
