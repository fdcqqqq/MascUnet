from keras.models import *
from keras.layers import *
from nets.attention_modules.attention_module import *



def res_block_atten(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    atten = BatchNormalization()(res_path)
    masc = Masc(atten)
    res_path = add([res_path, atten, masc])
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path
# F(x) = f(x) + x
def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path



def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)  # 第二层
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

    shortcut = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])
    skip = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), dilation_rate=(2, 2))(main_path)
    # skip = Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), dilation_rate=(2, 2))(skip)
    skip = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(16, 16))(skip)

    to_decoder.append(main_path)

    main_path = res_block_atten(main_path, [32, 32], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block_atten(main_path, [64, 64], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    main_path = res_block_atten(main_path, [128, 128], [(2, 2), (1, 1)])
    to_decoder.append(main_path)

    return to_decoder, skip


def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)
    main_path = concatenate([main_path, from_encoder[3]], axis=3)
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[2]], axis=3)
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)
    main_path = res_block(main_path, [32, 32], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)
    main_path = res_block(main_path, [16, 16], [(1, 1), (1, 1)])

    return main_path


def unet_model(inputs):


    to_decoder, skip = encoder(inputs)


    res_path = BatchNormalization()(to_decoder[3])
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(2, 2))(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(1, 1))(res_path)
    shortcut = Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2))(to_decoder[3])
    shortcut = BatchNormalization()(shortcut)

    path_all = add([shortcut, res_path])
    path_all = concatenate([path_all, skip], name='skip')
    path = decoder(path_all, from_encoder=to_decoder)

    path = Conv2D(filters=4, kernel_size=(1, 1))(path)
    output = Softmax()(path)
    model = Model(input=inputs, output=output)

    return model


if __name__ == '__main__':
    inputs = Input(shape=(128, 128, 4))
    model = unet_model(inputs)
    model.summary()
