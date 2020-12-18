from keras import backend as K
from nets.attention import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from layers import ConvOffset2D

K.set_image_dim_ordering("tf")
K.tensorflow_backend._get_available_gpus()


def relu6(x):
    return K.relu(x, max_value=6)


def conv3x3(x, out_filters, strides=(1, 1)):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(x)
    return x


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', use_activation=True):
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    if use_activation:
        x = Activation('relu')(x)
        return x
    else:
        return x

#
def basic_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = conv3x3(input, out_filters, strides)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # x = ConvOffset2D(out_filters)(x)

    x = conv3x3(x, out_filters)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def bottleneck_Block(input, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)


    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    # x = ConvOffset2D(out_filters)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, input])

    x = Activation('relu')(x)
    return x


def unet_model(inputs):
    # input = Input(shape=(height, width, channel))

    conv1_1 = Conv2D(32, 3, strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    conv1_1 = Conv2D(32, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(conv1_1)
    conv1_1 = BatchNormalization(axis=3)(conv1_1)
    conv1_1 = Activation('relu')(conv1_1)
    # conv1_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)

    # conv2_x  1/4
    conv2_1 = bottleneck_Block(conv1_1, 64, strides=(2, 2), with_conv_shortcut=True)
    conv2_2 = bottleneck_Block(conv2_1, 64)
    conv2_3 = bottleneck_Block(conv2_2, 64)

    # conv3_x  1/8
    conv3_1 = bottleneck_Block(conv2_3, 128, strides=(2, 2), with_conv_shortcut=True)
    conv3_2 = bottleneck_Block(conv3_1, 128)
    conv3_3 = bottleneck_Block(conv3_2, 128)
    conv3_4 = bottleneck_Block(conv3_3, 128)

    # conv4_x  1/16
    conv4_1 = bottleneck_Block(conv3_4, 256, strides=(2, 2), with_conv_shortcut=True)
    conv4_2 = bottleneck_Block(conv4_1, 256)
    conv4_3 = bottleneck_Block(conv4_2, 256)
    conv4_4 = bottleneck_Block(conv4_3, 256)
    conv4_5 = bottleneck_Block(conv4_4, 256)
    conv4_6 = bottleneck_Block(conv4_5, 256)
    conv4_7 = bottleneck_Block(conv4_6, 256)
    conv4_8 = bottleneck_Block(conv4_7, 256)
    conv4_9 = bottleneck_Block(conv4_8, 256)
    conv4_10 = bottleneck_Block(conv4_9, 256)
    conv4_11 = bottleneck_Block(conv4_10, 256)
    conv4_12 = bottleneck_Block(conv4_11, 256)
    conv4_13 = bottleneck_Block(conv4_12, 256)
    conv4_14 = bottleneck_Block(conv4_13, 256)
    conv4_15 = bottleneck_Block(conv4_14, 256)
    conv4_16 = bottleneck_Block(conv4_15, 256)
    conv4_17 = bottleneck_Block(conv4_16, 256)
    conv4_18 = bottleneck_Block(conv4_17, 256)
    conv4_19 = bottleneck_Block(conv4_18, 256)
    conv4_20 = bottleneck_Block(conv4_19, 256)
    conv4_21 = bottleneck_Block(conv4_20, 256)
    conv4_22 = bottleneck_Block(conv4_21, 256)
    conv4_23 = bottleneck_Block(conv4_22, 256)

    # conv5_x  1/32
    conv5_1 = bottleneck_Block(conv4_23, 512, strides=(2, 2), with_conv_shortcut=True)
    conv5_2 = bottleneck_Block(conv5_1, 512)
    conv5_3 = bottleneck_Block(conv5_2, 512)

    up6 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv5_3), 256, 3)
    merge6 = concatenate([conv4_23, up6], axis=3)
    conv6 = Conv2d_BN(merge6, 256, 3)
    conv6 = Conv2d_BN(conv6, 256, 3)

    up7 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv6), 128, 3)
    merge7 = concatenate([conv3_4, up7], axis=3)
    conv7 = Conv2d_BN(merge7, 128, 3)
    conv7 = Conv2d_BN(conv7, 128, 3)

    up8 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv7), 64, 3)
    merge8 = concatenate([conv2_3, up8], axis=3)
    conv8 = Conv2d_BN(merge8, 64, 3)
    conv8 = Conv2d_BN(conv8, 64, 3)

    up9 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv8), 32, 3)
    merge9 = concatenate([conv1_1, up9], axis=3)
    conv9 = Conv2d_BN(merge9, 32, 3)
    conv9 = Conv2d_BN(conv9, 32, 3)

    up10 = Conv2d_BN(UpSampling2D(size=(2, 2))(conv9), 32, 3)
    conv10 = Conv2d_BN(up10, 32, 3)
    conv10 = Conv2d_BN(conv10, 32, 3)

    conv11 = Conv2d_BN(conv10, 4, 1, use_activation=False)
    conv_softmax = Softmax()(conv11)

    model = Model(inputs=input, outputs=conv_softmax)

    # print(model.output_shape) compounded_loss
    # model_dice=dice_p_bce
    # model_dice=compounded_loss(smooth=0.0005,gamma=2., alpha=0.25)
    # model_dice=tversky_coef_loss_fun(alpha=0.3,beta=0.7)
    # model_dice=dice_coef_loss_fun(smooth=1e-5)
    # model.compile(optimizer = Nadam(lr = 2e-4), loss = model_dice, metrics = ['accuracy'])
    # 不使用metric
    # model_dice=focal_loss(alpha=.25, gamma=2)

    # model.compile(optimizer = Adam(lr = 2e-5),loss=dice_coef,metrics=['accuracy'])
    # model.compile(optimizer=Nadam(lr=2e-5), loss=focal_lossm, metrics=['accuracy'])

    # model.compile(optimizer = Nadam(lr = 2e-4), loss = "categorical_crossentropy",metrics=['accuracy'])
    return model


if __name__ == '__main__':
    input = Input(shape=(128, 128, 4))
    # model = unet_model(inputs=input)
    model = unet_model(inputs=input)
    model.summary()
