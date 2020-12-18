from __future__ import print_function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization, Add, Flatten, Dense, Dropout
from keras import backend as K

K.set_image_dim_ordering("tf")
K.tensorflow_backend._get_available_gpus()


def fcn_8s(inputs):
    x = Conv2D(32, (3, 3), activation=None, padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation=None, padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    p1 = x
    # Block 2
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), activation=None, padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    p2 = x

    # Block 3
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), activation=None, padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    p3 = x

    # Block 4
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    p4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), activation=None, padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    p5 = x

    # vgg = Model(inputs, x)

    ### pool4 (16,32,256) --> up1 (32,64,256)
    o = p4
    o = Conv2D(256, (5, 5), activation=None, padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2D(256, (3, 3), activation=None, padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same')(o)

    ### pool3 (32,64,128) --> (32,64,256)
    o2 = p3
    o2 = Conv2D(256, (3, 3), activation=None, padding='same')(o2)
    o2 = BatchNormalization()(o2)
    o2 = Activation('relu')(o2)

    ### concat1 [(32,64,256), (32,64,256)]
    o = concatenate([o, o2], axis=3)
    o = Conv2D(256, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### up2 (32,64,512) --> (64,128,128)
    o = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(o)
    o = Conv2D(128, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### pool2 (64,128,64) --> (64,128,128)
    o2 = p2
    o2 = Conv2D(128, (3, 3), activation=None, padding='same')(o2)
    o2 = BatchNormalization()(o2)
    o2 = Activation('relu')(o2)

    ### concat2 [(64,128,128), (64,128,128)]
    o = concatenate([o, o2], axis=3)

    ### up3 (64,128,256) --> (128,256,64)
    o = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(o)
    o = Conv2D(64, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### pool1 (128,256,64) --> (128,256,32)
    o2 = p1
    o2 = Conv2D(32, (3, 3), activation=None, padding='same')(o2)
    o2 = BatchNormalization()(o2)
    o2 = Activation('relu')(o2)

    ### concat3 [(128,256,64), (128,256,32)] --> (128,256,32)
    o = concatenate([o, o2], axis=3)
    o = Conv2D(32, (3, 3), padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    ### up (128,256,32) --> (256,512,32)
    o = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')(o)
    ### mask out (128,256,32) --> (256,512,3)
    o = Conv2D(4, (3, 3), padding='same')(o)
    o = Activation('softmax')(o)

    model = Model(inputs=[inputs], outputs=[o])

    return model


if __name__ == '__main__':
    inputs = Input(shape=(128, 128, 4))
    model = fcn_8s(inputs)
    model.summary()
