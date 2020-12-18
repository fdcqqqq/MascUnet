from keras.layers import Input
from keras.optimizers import SGD, Adam
from losses import *
from losses_f import *
# from nets.segnet_vgg import convnet_segnet
# from nets.segnet_res import resnet50_segnet
# from nets.deeplabV3 import Deeplabv3

# from nets.unet import unet_model
# from nets.unet_attention_sc import unet_model
# from nets.unet_attention_sc import unet_model
# from nets.unet_resnet_atten import unet_model
# from nets.masc_unet import unet_model
from nets.fdcunet1 import unet_model
# from nets.unet_res import unet_model

# from nets.unet_resnet_atten import unet_model
# from nets.fdcunet3 import unet_model
# from nets.unet import unet_model
# from nets.fdcunet1 import unet_model
# from nets.fdcunet2 import unet_model
# from nets.denseunet import unet_model
# from nets.atten_unet_sigmoid import unet_model
# from nets.unet_plus_plus import unet_model
# from nets.fcn8s import fcn_8s
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
        # #add gaussian noise to the first layer to combat overfitting
        #         i_=GaussianNoise(0.01)(i)

        #         i_ = Conv2D(64, 2, padding='same',data_format = 'channels_last')(i_)
        #         out=self.unet(inputs=i_)
        #         model = Model(input=i, output=out)
        # model = convnet_segnet(inputs=i)
        # model, _ = unet_model(inputs=i)
        model = unet_model(inputs=i)
        # model = fcn_8s(i)
        # model = resnet50_segnet(inputs=i)
        #         model = Deeplabv3(i, alpha=1.)
        # model = Deeplabv3(i, alpha=1., OS=16)
        #         model = get_resnet50_encoder(inputs=i)
        model.summary()

        #         adam = Adam(lr=0.001)
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=5e-6, nesterov=False),
                      loss=gen_dice_loss,
                      metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
        # model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=5e-6, nesterov=False),
        #               loss={'vae_pre': 'mse', 'unet_pre': gen_dice_loss},
        #               loss_weights={'vae_pre': 1., 'unet_pre': 1.},
        #               metrics={'vae_pre': 'mse', 'unet_pre': [dice_whole_metric, dice_core_metric, dice_en_metric]})
        # load weights if set for prediction
        if self.load_model_weights is not None:
            model.load_weights(self.load_model_weights)
        return model
