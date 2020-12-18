from keras.models import *
from keras.layers import *
from nets.convnet import get_convnet_encoder
import tensorflow as tf

IMAGE_ORDERING = 'channels_last'


def segnet_decoder(f, n_classes, n_up=3):
    assert n_up >= 2

    o = f
    # 26,26,512
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/8
    # 52,52,256
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 104,104,128
    for _ in range(n_up - 2):
        o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
        o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
        o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
        o = (BatchNormalization())(o)

    # 进行一次UpSampling2D，此时hw变为原来的1/2
    # 208,208,64
    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    # 此时输出为h_input/2,w_input/2,nclasses
    # 208,208,2
    o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

    return o


def _segnet(inputs, encoder):
    # encoder通过主干网络
    img_input, levels = encoder(inputs)

    # 获取hw压缩四次后的结果
    feat = levels[3]

    # 将特征传入segnet网络
    o = segnet_decoder(feat, 4, n_up=3)

    size_before3 = tf.keras.backend.int_shape(inputs)
    o = Lambda(lambda xx: tf.image.resize_images(xx, size_before3[1:3]))(o)
    o = Softmax()(o)
    model = Model(img_input, o)

    return model


def convnet_segnet(inputs):
    model = _segnet(inputs, get_convnet_encoder)
    model.model_name = "convnet_segnet"
    return model


if __name__ == '__main__':
    inputs = Input(shape=(128, 128, 4))
    model = convnet_segnet(inputs)
    model.summary()
