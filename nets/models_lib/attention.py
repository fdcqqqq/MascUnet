from functools import wraps
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
# from tools.utils import compose
from keras.activations import relu, softmax
from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Input
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, \
    multiply, Permute, Concatenate, Conv2D, Add, Activation, \
    Lambda, BatchNormalization, add, Input
from keras import backend as K
# from  keras.utils import
from keras.layers import Input, concatenate, UpSampling2D
from keras.models import Model
import numpy as np

import tensorflow as tf


# 注意力机制

def channel_attention(input_feature, ratio):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])  # 组合池化方式一
    cbam_feature = Activation('softmax')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 3

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])  # 组合池化方式二
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='softmax',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio):
    """
    channel & spatial 的串行结构
    :param cbam_feature:
    :param ratio:
    :return:
    """
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def squeeze(inputs):
    # 注意力机制单元
    input_channels = int(inputs.shape[-1])

    # x = GlobalAveragePooling2D()(inputs)
    x = adap_maxpooling(inputs, 1)
    x = Dense(int(input_channels / 4))(x)
    x = Activation(relu)(x)
    x = Dense(input_channels)(x)
    x = Activation(softmax)(x)
    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])

    return x


# attention unet
def attention_up_and_concate(down_layer, layer):
    up = UpSampling2D(size=(2, 2))(down_layer)
    layer = attention_block_2d(x=layer, g=up)
    concate = concatenate([up, layer], axis=3)
    return concate


def attention_block_2d(x, g):
    inter_channel = int(x.shape[-1]) // 4
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1])(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1])(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1])(f)
    rate = Activation(softmax)(psi_f)
    att_x = multiply([x, rate])

    return att_x


def CSAR(input, reduction=8, increase=2):
    """
    channel & spatial attention 的并行结构
    :param input:
    :param reduction:
    :param increase:
    :return:
    """
    channel = int(input.shape[-1])  # (B, W, H, C)
    #
    # u = Conv2D(channel, 3, padding='same')(input)  # (B, W, H, C)
    # u = Activation('relu')(u)
    # u = Conv2D(channel, 3, padding='same')(u)  # (B, W, H, C)

    # channel attention
    # x = GlobalAveragePooling2D()(u)
    # 全局平均池化
    x = Lambda(lambda x: K.mean(input, axis=(1, 2), keepdims=True))(input)  # (B, 1, 1, C)
    x = Conv2D(channel // reduction, 1)(x)  # (B, 1, 1, C // r)
    x = Activation('relu')(x)
    x = Conv2D(channel, 1)(x)  # (B, 1, 1, C)
    x = Activation(softmax)(x)
    x = multiply([input, x])  # (B, W, H, C)

    # # spatial attention
    # y = Conv2D(channel * increase, 1)(u)  # (B, W, H, C * i)
    # y = Activation('relu')(y)
    # y = Conv2D(1, 1)(y)  # (B, W, H, 1)
    # y = Activation(softmax)(y)
    # y = multiply([u, y])  # (B, W, H, C)
    # spatial attention
    # y = DepthwiseConv2D(channel, 1, padding='same')(input)  # (B, W, H, C * i)
    y = Conv2D(channel * increase, 1)(input)  # (B, W, H, C * i)
    y = Activation('relu')(y)
    y = Conv2D(1, 1)(y)  # (B, W, H, 1)
    y = Activation(softmax)(y)
    s1 = multiply([input, y])  # (B, W, H, C)

    # z = concatenate([x, y], -1)
    # z = Conv2D(channel, 1)(z)  # (B, W, H, C)
    # s1 = Activation('relu')(z)
    # z = add([input, z])

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
    concat = Concatenate(axis=3)([avg_pool, max_pool])  # 组合池化方式二
    cbam_feature = Conv2D(filters=1,
                          kernel_size=3,
                          activation='softmax',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    s2 = multiply([input, cbam_feature])
    out_feature = add([s1, s2])
    return out_feature


def adap_maxpooling(x, outsize):
    # x_shape = K.int_shape(x)
    # batchsize1, dim1, dim2, channels1 = x_shape
    dim = int(x.shape[1])
    # np.floor():返回不大于输入参数的最大整数。（向下取整）
    stride = np.floor(dim / outsize).astype(np.int32)
    kernels = dim - (outsize - 1) * stride
    adap_pooling = MaxPooling2D(pool_size=(kernels, kernels), strides=(stride, stride))(x)

    return adap_pooling


def RAM(input, channels, reduction_ratio=16,kernel_size = 3):
    # pre-attention feature extraction
    x = Conv2D(channels, kernel_size, strides=1, padding='same')(input)
    x = Activation('relu')(x)
    x = Conv2D(channels, kernel_size, strides=1, padding='same')(x)

    # compute attentions
    _, ca = Lambda(lambda x: tf.nn.moments(x, axes=[1, 2]))(input)
    ca = Dense(channels // reduction_ratio)(ca)
    ca = Activation('relu')(ca)
    ca = Dense(channels)(ca)

    sa = DepthwiseConv2D(kernel_size, padding='same')(input)
    fa = add([ca, sa])
    fa = Activation('softmax')(fa)
    # apply attention
    x = multiply([x, fa])

    return add([input, x])

if __name__ == '__main__':
    pass
