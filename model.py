from keras.layers import Input
from keras.optimizers import SGD, Adam
from losses import *
from nets.MascParallelUnet import unet_model
K.set_image_data_format("channels_last")


# u-net model
class Unet_model(object):

    def __init__(self, img_shape, load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        self.model = self.compile_unet()

    def compile_unet(self):
        """
        compile the U-net model
        """
        i = Input(shape=self.img_shape)
        model = unet_model(inputs=i)
        model.summary()

        adam = Adam(lr=0.001)
        sgd = SGD(lr=0.001, momentum=0.9, decay=5e-6, nesterov=False)
        model.compile(optimizer=adam,
                      loss=gen_dice_loss,
                      metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])

        # load weights if set for prediction
        if self.load_model_weights is not None:
            model.load_weights(self.load_model_weights)
        return model
