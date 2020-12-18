from keras.layers import *
from nets.attention import *



# 基础模型的设置
# 总共22层卷积=shortcut(7层)+res_block(15层)
def res_block_atten(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)

    atten = BatchNormalization()(res_path)
    cbam = cbam_block(atten, ratio=8)
    res_path = add([res_path, atten, cbam])
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])  # 残差块结尾相加
    return res_path
# 定义残差块函数,核心思想: F(x) = f(x) + x
def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])  # 残差块结尾相加
    return res_path
# 定义下采样函数
def encoder(x):
    to_decoder = []

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(x)  # 第二层
    main_path = BatchNormalization()(main_path)
    main_path = Activation(activation='relu')(main_path)

    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)
    shortcut = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1))(x)
    shortcut = BatchNormalization()(shortcut)

    main_path = add([shortcut, main_path])#conv1第一层
    # (encoder)第二层数据进入第八层(decoder)
    to_decoder.append(main_path)

    main_path = res_block_atten(main_path, [64, 64], [(2, 2), (1, 1)])  # 第三层([128,128]是两层卷积核的个数,2代表卷积核大小,1代表步长)
    to_decoder.append(main_path)

    main_path = res_block_atten(main_path, [128, 128], [(2, 2), (1, 1)])  # 第四层
    to_decoder.append(main_path)

    main_path = res_block_atten(main_path, [256, 256], [(2, 2), (1, 1)])  # 第四层
    to_decoder.append(main_path)

    return to_decoder


# 定义上采样函数
def decoder(x, from_encoder):
    main_path = UpSampling2D(size=(2, 2))(x)  # 2*2卷积核上采样
    main_path = concatenate([main_path, from_encoder[2]], axis=3)  # 第六层与第四层跳接
    main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])  # 256卷积核个数;1卷积核大小,1步长

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[1]], axis=3)  # 第七层和第三层跳接
    main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

    main_path = UpSampling2D(size=(2, 2))(main_path)
    main_path = concatenate([main_path, from_encoder[0]], axis=3)  # 第八层和第二层跳接
    main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

    return main_path


# 定义残差块和unet结合的函数
def unet_model(inputs):
    # inputs = Input(shape=input_shape)

    to_decoder = encoder(inputs)
    path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])  # 第五层
    path = decoder(path, from_encoder=to_decoder)

    path = Conv2D(filters=4, kernel_size=(1, 1))(path)  # 第九层:输出层
    output = Softmax()(path)
    model = Model(input=inputs, output=output)

    return model
if __name__ == '__main__':
    inputs = Input(shape=(128,128,4))
    model = unet_model(inputs)
    model.summary()