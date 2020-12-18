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
    atten1 = BatchNormalization()(conv1)
    masc1 = CSAR(atten1)
    conv1 = add([conv1, atten1, masc1])
    conv1 = Conv2D(32, 3, 1, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(pool1)
    atten2 = BatchNormalization()(conv2)
    masc2 = CSAR(atten2)
    conv2 = add([conv2, atten2, masc2])
    conv2 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(pool2)
    atten3 = BatchNormalization()(conv3)
    masc3 = CSAR(atten3)
    conv3 = add([conv3, atten3, masc3])
    conv3 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(pool3)
    atten4 = BatchNormalization()(conv4)
    masc4 = CSAR(atten4)
    conv4 = add([conv4, atten4, masc4])
    conv4 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, 3, 1, activation='relu', border_mode='same')(pool4)
    conv5 = Conv2D(512, 3, 1, activation='relu', border_mode='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(up6)
    conv6 = Conv2D(256, 3, 1, activation='relu', border_mode='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(up7)
    conv7 = Conv2D(128, 3, 1, activation='relu', border_mode='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(up8)
    conv8 = Conv2D(64, 3, 1, activation='relu', border_mode='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, 3, 1, activation='relu', border_mode='same')(up9)
    conv9 = Conv2D(32, 3, 1, activation='relu', border_mode='same')(conv9)

    conv10 = Conv2D(4, 1, 1, activation='softmax')(conv9)

    model = Model(input=inputs, output=conv10)

    # model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


if __name__ == '__main__':
    input = Input(shape=(128, 128, 4))
    model = unet_model(inputs=input)
    model.summary()
