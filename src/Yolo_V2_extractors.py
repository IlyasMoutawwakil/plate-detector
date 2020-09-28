from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Reshape, GlobalMaxPool2D, MaxPooling2D, concatenate
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Input, Dropout, DepthwiseConv2D
from tensorflow.keras.layers import LeakyReLU, ReLU, GlobalAveragePooling2D, GlobalMaxPool2D, Reshape, Activation
import tensorflow as tf

<<<<<<< HEAD

ROOT_DIR = 'C:/Users/utilisateur/Documents/GitHub/plate-detector/'
BACKEND_DIR = ROOT_DIR + 'backends/'
SQUEEZENET_BACKEND_PATH = BACKEND_DIR + "squeezenet_backend.h5"
MOBILENET_BACKEND_PATH  = BACKEND_DIR + "mobilenet_backend.h5"
=======
ROOT_DIR = "/gdrive/My Drive/Colab Notebooks/plate-detector/"
BACKEND_DIR = ROOT_DIR + 'backends/'

SQUEEZENET_BACKEND_PATH = BACKEND_DIR + "squeezenet_backend.h5"  # should be hosted on the server
MOBILENET_BACKEND_PATH  = BACKEND_DIR + "mobilenet_backend.h5"   # should be hosted on the server
>>>>>>> 465d6c8f4f035fb2ee5ab96aa3f139dd0a62fcce

class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")

    def get_output_shape(self):
        return self.feature_extractor.get_output_shape_at(-1)[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

class FullYoloFeature(BaseFeatureExtractor):
    '''Le classifieur DarkNet développé originalement pour être utilisé avec l'algorithme de détéction YOLO'''

    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3)) # On peut choisir comme comme nombre de channel
                                                               # soit 1 (images grises) ou 3 (images RGB)

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            return tf.space_to_depth(x, block_size=2)

        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)
        self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)

    def normalize(self, image):
        return image / 255.



class MobileNetFeature(BaseFeatureExtractor):

    def __init__(self, input_size):

        width_multiplier = 1  # Changes number of filters
        depth_multiplier = 1  # Resolution Multiplier
        pooling = None        # Global Average/Max Pooling or None

        def conv_block(inputs,filters,width_multiplier,kernel_size=(3,3),strides=(1,1)):
            filters = int(filters*width_multiplier)
            x = ZeroPadding2D(padding=((0,1),(0,1)),name='conv1_pad')(inputs)
            x = Conv2D(filters,kernel_size,padding='valid',use_bias=False,strides=strides,name='conv1')(x)
            x = BatchNormalization(name='conv1_bn')(x)
            x = ReLU(6.,name='conv1_relu')(x)
            return x


        def mobile_block(inputs,pointwise_conv_filters,width_multiplier,depth_multiplier=1,strides=(1,1),block_id=1):
            pointwise_conv_filters = int(pointwise_conv_filters*width_multiplier)

            if strides==(1,1):
                x = inputs
            else:
                x = ZeroPadding2D(padding=((0,1),(0,1)),name='conv_pad_%d'%block_id)(inputs)

            x = DepthwiseConv2D((3,3),padding='same' if strides==(1,1) else 'valid',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,name='conv_dw_%d'%block_id)(x)
            x = BatchNormalization(name='conv_dw_%d_bn'%block_id)(x)
            x = ReLU(6.,name='conv_dw_%d_relu'%block_id)(x)

            # PointWise Convolution with 1X1 Filters, No of Filters = pointwise_conv_filters
            x = Conv2D(pointwise_conv_filters,(1,1),padding='same',use_bias=False,strides=(1,1),name='conv_pw_%d'%block_id)(x)
            x = BatchNormalization(name='conv_pw_%d_bn'%block_id)(x)
            x = ReLU(6.,name='conv_pw_%d_relu'%block_id)(x)
            return x

        input_image = Input(shape=(input_size, input_size, 3))
        x = conv_block(input_image,32,width_multiplier,strides=(2,2))

        #Block 1
        x = mobile_block(x,64,width_multiplier,depth_multiplier,block_id=1)
        #Block2
        x = mobile_block(x,128,width_multiplier,depth_multiplier,strides=(2,2),block_id=2)
        #Block 3
        x = mobile_block(x,128,width_multiplier,depth_multiplier,block_id=3)
        #Block 4
        x = mobile_block(x,256,width_multiplier,depth_multiplier,strides=(2,2),block_id=4)
        #Block 5
        x = mobile_block(x,256,width_multiplier,depth_multiplier,block_id=5)
        #Block 6
        x = mobile_block(x,512,width_multiplier,depth_multiplier,strides=(2,2),block_id=6)
        #Block 7
        x = mobile_block(x,512,width_multiplier,depth_multiplier,block_id=7)
        #Block 8
        x = mobile_block(x,512,width_multiplier,depth_multiplier,block_id=8)
        #Block 9
        x = mobile_block(x,512,width_multiplier,depth_multiplier,block_id=9)
        #Block 10
        x = mobile_block(x,512,width_multiplier,depth_multiplier,block_id=10)
        #Block 11
        x = mobile_block(x,512,width_multiplier,depth_multiplier,block_id=11)
        #Block 12
        x = mobile_block(x,1024,width_multiplier,depth_multiplier,strides=(2,2),block_id=12)
        #Block 13
        x = mobile_block(x,1024,width_multiplier,depth_multiplier,block_id=13)

        if pooling=='avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)

        self.feature_extractor = Model(inputs=input_image,outputs=x,name='mobilenet')
        self.feature_extractor.load_weights(MOBILENET_BACKEND_PATH)

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.
        return image

class SqueezeNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_size):

        # define some auxiliary variables and the fire module
        sq1x1  = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu   = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x     = Activation('relu', name=s_id + relu + sq1x1)(x)

            left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left  = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=(input_size, input_size, 3))

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = Model(inputs=input_image, outputs=x, name='squeezenet')
        self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')
        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68
        return image
