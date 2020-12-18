import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.transform as trans
import random as r
from keras.models import Sequential, load_model, Model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, MaxPool2D
from keras.layers import Input, concatenate, UpSampling2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
from nets.attention import *

K.set_image_dim_ordering("tf")
K.tensorflow_backend._get_available_gpus()


def unet_model(inputs):
    # inputs = Input((128, 128, 4))
    conv1 = Conv2D(32, 3, 1, activation='relu', border_mode='same')(inputs)  # KERNEL =3 STRIDE =3
    conv1 = Conv2D(32, 3, 1, activation='relu', border_mode='same')(conv1)
    atten1 = BatchNormalization()(conv1)
    cbam1 = cbam_block(atten1, ratio=8)
    attention1 = add([conv1, atten1, cbam1])
    pool1 = MaxPooling2D(pool_size=(2, 2))(attention1)

    conv2 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(pool1)
    conv2 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(conv2)
    atten2 = BatchNormalization()(conv2)
    cbam2 = cbam_block(atten2, ratio=8)
    attention2 = add([conv2, atten2, cbam2])
    pool2 = MaxPooling2D(pool_size=(2, 2))(attention2)

    conv3 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(pool2)
    conv3 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(conv3)
    atten3 = BatchNormalization()(conv3)
    cbam3 = cbam_block(atten3, ratio=8)
    attention3 = add([conv3, atten3, cbam3])
    pool3 = MaxPooling2D(pool_size=(2, 2))(attention3)

    conv4 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(pool3)
    conv4 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(conv4)
    atten4 = BatchNormalization()(conv4)
    cbam4 = cbam_block(atten4, ratio=8)
    attention4 = add([conv4, atten4, cbam4])
    pool4 = MaxPooling2D(pool_size=(2, 2))(attention4)

    conv5 = Conv2D(512, 3, 1, activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, 3, 1, activation='relu', border_mode='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(conv6)
    atten6 = BatchNormalization()(conv6)
    cbam6 = cbam_block(atten6, ratio=8)
    attention6 = add([conv6, atten6, cbam6])

    up7 = concatenate([UpSampling2D(size=(2, 2))(attention6), conv3], axis=3)
    conv7 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(conv7)
    atten7 = BatchNormalization()(conv7)
    cbam7 = cbam_block(atten7, ratio=8)
    attention7 = add([conv7, atten7, cbam7])

    up8 = concatenate([UpSampling2D(size=(2, 2))(attention7), conv2], axis=3)
    conv8 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(conv8)
    atten8 = BatchNormalization()(conv8)
    cbam8 = cbam_block(atten8, ratio=8)
    attention8 = add([conv8, atten8, cbam8])

    up9 = concatenate([UpSampling2D(size=(2, 2))(attention8), conv1], axis=3)
    conv9 = Conv2D(32, 3, 1, activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, 3, 1, activation='relu', border_mode='same')(conv9)
    atten9 = BatchNormalization()(conv9)
    cbam9 = cbam_block(atten9, ratio=8)
    attention9 = add([conv9, atten9, cbam9])

    conv10 = Conv2D(4, 1, 1, activation='softmax')(attention9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


if __name__ == '__main__':
    input = Input(shape=(128, 128, 4))
    model = unet_model(inputs=input)
    model.summary()
