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
				
import scipy.io as sio
# from TrActLayer import trAct_1D_Exp, trAct_2D_Exp
nb_classes = 10
epochs = 200


import scipy.io as sio
import copy

for trial in range(1):


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
    # x_train2 -= x_train_mean

    x_train_C -= x_train_mean
    x_train_W -= x_train_mean
    x_test_C-= x_train_mean
    x_test_W -= x_train_mean
    # x_test2 -= x_train_mean




    print(np.max(x_train))



    # x_train = np.concatenate((x_train,x_train3),axis=0)
    # y_train = np.concatenate((y_train,y_train3),axis=0)

    # (X_train2, y_train2), (X_test2, y_test2) = mnist.load_data()
    # x_train = x_train.astype('float32') / 255.



    # X_train2 = X_train2.astype('float32')
    # X_train2 /= 255.
    # X_train2 = np.reshape(X_train2,[60000,28,28,1])

    import copy
    # If subtract pixel mean is enabled



    # Convert class vectors to binary class matrices.
    y_train = tensorflow.keras.utils.to_categorical(y_train_C, nb_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test_C, nb_classes)
    y_train2 = tensorflow.keras.utils.to_categorical(y_train2, nb_classes)

    y_test_W = tensorflow.keras.utils.to_categorical(y_test_W, nb_classes)
    y_test2 = tensorflow.keras.utils.to_categorical(y_test2, nb_classes)

    lll = 0.0
    NumLength=5

    trainableLabel = True
    def lr_schedule(epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
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
        # x = Activation('relu')(x)

        # x = Conv2D(filters3, (1, 1),
                          # kernel_initializer='he_normal',
                          # name=conv_name_base + '2c')(x)
        # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

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
        # x = Activation('relu')(x)

        # x = Conv2D(filters3, (1, 1),
                          # kernel_initializer='he_normal',
                          # name=conv_name_base + '2c')(x)
        # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

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
        # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = identity_block(x, 3, [64, 64], stage=2, block='a')
        x = identity_block(x, 3, [64, 64], stage=2, block='b')
        # x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128], stage=3, block='a', strides=(1, 1))
        x = identity_block(x, 3, [128, 128], stage=3, block='b')
        # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        # x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256], stage=4, block='a')
        x = identity_block(x, 3, [256, 256], stage=4, block='b')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        # x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512], stage=5, block='a')
        x = identity_block(x, 3, [512, 512], stage=5, block='b')
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(10, activation='softmax', name='fc1000')(x)
        model = Model(img_input, x, name='resnet18')


        return model

                    
    datagen = ImageDataGenerator(
            # randomly rotate images in the range (deg 0 to 180)
            rotation_range=10,
            # randomly shift images horizontally
            width_shift_range=0.15,
            # randomly shift images vertically
            height_shift_range=0.15,
            horizontal_flip = True,
            # set range for random shear
            shear_range=0.1,
            # set range for random zoom
            zoom_range=0.3,
            # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            # value used for fill_mode = "constant"
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
    # training
    TRAINING_BATCH_SIZE      = 32
    TRAINING_SHUFFLE_BUFFER  = 5000
    TRAINING_BN_MOMENTUM     = 0.99
    TRAINING_BN_EPSILON      = 0.001
    TRAINING_LR_MAX          = 0.001
    # TRAINING_LR_SCALE        = 0.1
    # TRAINING_LR_EPOCHS       = 2
    TRAINING_LR_INIT_SCALE   = 0.01
    TRAINING_LR_INIT_EPOCHS  = 10
    TRAINING_LR_FINAL_SCALE  = 0.01
    TRAINING_LR_FINAL_EPOCHS = 100
    # training (derived)
    TRAINING_NUM_EPOCHS = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
    TRAINING_LR_INIT    = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
    TRAINING_LR_FINAL   = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE    
    model = ResNet18()

    # model = multi_gpu_model(model_s,2)                
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    batch_size = 512
    callbacks = [lr_scheduler]
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])	
    model.summary()



    Loss = []
    Acc = []
    ValLoss = []
    ValAcc = []
    cnt = 1

    # for kk in range(10):  
        # for ii in range(NumLength):
            # print('Epoch : ', cnt)
            # cnt = cnt+1
    hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),steps_per_epoch=int(x_train.shape[0] / batch_size),epochs=100, verbose=1, workers=1, shuffle=True,validation_data = (x_test, y_test), callbacks=callbacks )
    # Loss.append(hist.history['loss'][-1])
    # ValLoss.append(hist.history['val_loss'][-1])
    # Acc.append(hist.history['acc'][-1])
    # ValAcc.append(hist.history['val_acc'][-1])

    output1 = model.evaluate(x_train_C,y_train)
    output2 = model.evaluate(x_train_W,y_train)
    output3 = model.evaluate(x_train2,y_train2)
    output4 = model.evaluate(x_test_C,y_test)    
    output5 = model.evaluate(x_test_W,y_test)
    output6 = model.evaluate(x_test2,y_test2)

    sio.savemat('Evaluate_Results.mat', mdict = {'Acc_clean_tr':output1[1], 'CE_clean_tr':output1[0], 'Acc_watermarked_tr':output2[1], 'CE_watermarked_tr':output2[0],'Acc_watermark_tr':output3[1], 'CE_watermark_tr':output3[0],  'Acc_clean_te':output4[1], 'CE_clean_te':output4[0], 'Acc_watermarked_te':output5[1], 'CE_watermarked_te':output5[0],'Acc_watermark_te':output6[1], 'CE_watermark_te':output6[0],  'ValAcc':hist.history['val_acc'], 'Loss':hist.history['loss'], 'ValLoss':hist.history['val_loss'], 'Acc':hist.history['acc'] })
    # model.save_weights('Weights_Watermarked_50Percent_Trial'+ str(trial)+ '_ResNet18_Version2_Long_CaseX1_3_FLIPLR.h5')
    K.clear_session()