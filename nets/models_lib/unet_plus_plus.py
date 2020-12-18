from keras.layers import MaxPooling2D, Conv2D, Conv2DTranspose, Input, concatenate
from keras.models import Model


def unet_model(inputs):
    conv0_0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv0_0 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0_0)

    conv1_0 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv1_0 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

    up1_0 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv1_0)
    merge00_10 = concatenate([conv0_0, up1_0], axis=-1)
    conv0_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge00_10)
    conv0_1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_1)

    conv2_0 = Conv2D(32 * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv2_0 = Conv2D(32 * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

    up2_0 = Conv2DTranspose(32 * 2, (2, 2), strides=(2, 2), padding='same')(conv2_0)
    merge10_20 = concatenate([conv1_0, up2_0], axis=-1)
    conv1_1 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge10_20)
    conv1_1 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)

    up1_1 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv1_1)
    merge01_11 = concatenate([conv0_0, conv0_1, up1_1], axis=-1)
    conv0_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge01_11)
    conv0_2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_2)

    conv3_0 = Conv2D(32 * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv3_0 = Conv2D(32 * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

    up3_0 = Conv2DTranspose(32 * 4, (2, 2), strides=(2, 2), padding='same')(conv3_0)
    merge20_30 = concatenate([conv2_0, up3_0], axis=-1)
    conv2_1 = Conv2D(32 * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge20_30)
    conv2_1 = Conv2D(32 * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)

    up2_1 = Conv2DTranspose(32 * 2, (2, 2), strides=(2, 2), padding='same')(conv2_1)
    merge11_21 = concatenate([conv1_0, conv1_1, up2_1], axis=-1)
    conv1_2 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge11_21)
    conv1_2 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_2)

    up1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv1_2)
    merge02_12 = concatenate([conv0_0, conv0_1, conv0_2, up1_2], axis=-1)
    conv0_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge02_12)
    conv0_3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_3)

    conv4_0 = Conv2D(32 * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv4_0 = Conv2D(32 * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        conv4_0)

    up4_0 = Conv2DTranspose(32 * 8, (2, 2), strides=(2, 2), padding='same')(conv4_0)
    merge30_40 = concatenate([conv3_0, up4_0], axis=-1)
    conv3_1 = Conv2D(32 * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge30_40)
    conv3_1 = Conv2D(32 * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)

    up3_1 = Conv2DTranspose(32 * 4, (2, 2), strides=(2, 2), padding='same')(conv3_1)
    merge21_31 = concatenate([conv2_0, conv2_1, up3_1], axis=-1)
    conv2_2 = Conv2D(32 * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge21_31)
    conv2_2 = Conv2D(32 * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_2)

    up2_2 = Conv2DTranspose(32 * 2, (2, 2), strides=(2, 2), padding='same')(conv2_2)
    merge12_22 = concatenate([conv1_0, conv1_1, conv1_2, up2_2], axis=-1)
    conv1_3 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        merge12_22)
    conv1_3 = Conv2D(32 * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_3)

    up1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv1_3)
    merge03_13 = concatenate([conv0_0, conv0_1, conv0_2, conv0_3, up1_3], axis=-1)
    conv0_4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge03_13)
    conv0_4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv0_4)

    conv0_4 = Conv2D(4, 1, activation='softmax')(conv0_4)

    model = Model(input=inputs, output=conv0_4)

    return model


if __name__ == '__main__':
    inputs = Input(shape=(128, 128, 4))
    model = unet_model(inputs)
    model.summary()
