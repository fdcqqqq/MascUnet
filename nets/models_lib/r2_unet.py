from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input, add, core
from keras.models import Model
from keras.layers.core import Lambda
import keras.backend as K


def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
                  padding='same'):
    input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding)(input_layer)
    else:
        skip_layer = input_layer
    layer = skip_layer
    for j in range(2):
        for i in range(2):
            if i == 0:
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer


def up_and_concate(down_layer, layer):
    # in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2))(down_layer)

    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate


def unet_model(inputs):
    x = inputs
    depth = 4
    features = 32
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        features = features * 2

    x = rec_res_block(x, features)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i])
        x = rec_res_block(x, features)

    conv6 = Conv2D(4, (1, 1), padding='same')(x)
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model


if __name__ == '__main__':
    inputs = Input(shape=(128, 128, 4))
    model = unet_model(inputs)
    model.summary()
