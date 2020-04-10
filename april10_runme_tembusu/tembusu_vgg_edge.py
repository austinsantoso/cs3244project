from keras.preprocessing.image import ImageDataGenerator
train_datagen =  ImageDataGenerator(rescale=1.0/255, 
                                    rotation_range=15,
                                    zoom_range=0.2)

val_datagen =  ImageDataGenerator(rescale=1.0/255)

test_datagen =  ImageDataGenerator(rescale=1.0/255)

"""
CHOOSE THE DIRECTORY OF IMAGE
"""
# picture_main_dir = '../train_copy/'
# picture_main_dir = '../custom_training_data/'
picture_main_dir = '../custom_training_data_edge/'

filename_extension= '_vgg_edge'

nrows = 100
ncolumns = 100
batch_size = 50


train_gen = train_datagen.flow_from_directory(directory= picture_main_dir + 'train/',
                                              target_size=(100, 100),
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              shuffle=False,
                                              seed=0)
val_gen = val_datagen.flow_from_directory(directory= picture_main_dir + 'val/',
                                          target_size=(100, 100),
                                          color_mode='rgb',
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          shuffle=False,
                                          seed=0)

test_gen = test_datagen.flow_from_directory(directory= picture_main_dir + 'test/',
                                            target_size=(100, 100),
                                            color_mode='rgb',
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False,
                                            seed=0)

"""
MODEL
====================================================================================
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

"""
CaffeNet
"""
def CaffeNet(input_shape=(100,100,1),nclass=24):
    #Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='same'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.5))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(nclass))
    model.add(Activation('softmax'))

    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    return model


"""
VGG19
====================================================================================
"""

def VGG19(input_shape=(100,100,1),nclass=24): 
    model = Sequential()

    # Block 1
    model.add(Conv2D(filters=64,
                     input_shape=input_shape,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # Block 2
    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # Block 3
    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=256,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # Block 4
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # Block 5
    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

    # Passing it to a Fully Connected layer
    model.add(Flatten())

    # 1st Fully Connected Layer
    model.add(Dense(4096, activation='relu'))

    # 2nd Fully Connected Layer
    model.add(Dense(4096, activation='relu'))

    # Output layer
    model.add(Dense(nclass, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    return model


"""
from_paper
====================================================================================
"""

def from_paper(input_shape=(100,100,1),nclass=24):
    model = Sequential()
    
    model.add(Conv2D(filters=16,
                     input_shape=input_shape,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    
    model.add(MaxPooling2D(pool_size=(5, 5)))
    
    model.add(Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    
    model.add(MaxPooling2D(pool_size=(5, 5)))
    
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    
    model.add(Dense(nclass, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    
    return model

"""
ResNext
====================================================================================
"""
from keras import backend, layers, models
from keras import utils as keras_utils
import numpy as np

def block3(x, filters, kernel_size=3, stride=1, groups=32,
           conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv')(x)
    kernel = np.zeros((1, 1, filters * c, filters), dtype=np.float32)
    for i in range(filters):
        start = (i // c) * c * c + i % c
        end = start + c * c
        kernel[:, :, start:end:c, i] = 1.
    x = layers.Conv2D(filters, 1, use_bias=False, trainable=False,
                      kernel_initializer={'class_name': 'Constant',
                                          'config': {'value': kernel}},
                      name=name + '_2_gconv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D((64 // groups) * filters, 1,
                      use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False,
                   name=name + '_block' + str(i))
    return x


def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights=None,
           input_tensor=None,
           input_shape=None,
           pooling=None,
           nclass=1000,
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        preact: whether to use pre-activation or not
            (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
            (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    # Determine proper input shape
#     input_shape = input_shape

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(nclass, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name=model_name)

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def ResNeXt50(include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=(100,100,1),
              pooling=None,
              classes=24,
              **kwargs):
    def stack_fn(x):
        x = stack3(x, 128, 3, stride1=1, name='conv2')
        x = stack3(x, 256, 4, name='conv3')
        x = stack3(x, 512, 6, name='conv4')
        x = stack3(x, 1024, 3, name='conv5')
        return x
    return ResNet(stack_fn, False, False, 'resnext50',
                  include_top, weights,
                  input_tensor, input_shape,
                  pooling, classes,
                  **kwargs)

"""
InceptionResNet2
====================================================================================
"""
from keras.layers import Concatenate, Lambda, Input, GlobalAveragePooling2D
from keras.models import Model


def conv2d(x,numfilt,filtsz,strides=1,pad='same',act=True,name=None):
  x = Conv2D(numfilt,filtsz,strides=strides,padding=pad,data_format='channels_last',use_bias=False,name=name+'conv2d')(x)
  x = BatchNormalization(axis=3,scale=False,name=name+'conv2d'+'bn')(x)
  if act:
    x = Activation('relu',name=name+'conv2d'+'act')(x)
  return x

def incresA(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,32,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,32,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,32,3,1,pad,True,name=name+'b1_2')
    branch2 = conv2d(x,32,1,1,pad,True,name=name+'b2_1')
    branch2 = conv2d(branch2,48,3,1,pad,True,name=name+'b2_2')
    branch2 = conv2d(branch2,64,3,1,pad,True,name=name+'b2_3')
    branches = [branch0,branch1,branch2]
    mixed = Concatenate(axis=3, name=name + '_concat')(branches)
    filt_exp_1x1 = conv2d(mixed,384,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresB(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,128,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,160,[1,7],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,192,[7,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,1152,1,1,pad,False,name=name+'filt_exp_1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_scaling')([x, filt_exp_1x1])
    return final_lay

def incresC(x,scale,name=None):
    pad = 'same'
    branch0 = conv2d(x,192,1,1,pad,True,name=name+'b0')
    branch1 = conv2d(x,192,1,1,pad,True,name=name+'b1_1')
    branch1 = conv2d(branch1,224,[1,3],1,pad,True,name=name+'b1_2')
    branch1 = conv2d(branch1,256,[3,1],1,pad,True,name=name+'b1_3')
    branches = [branch0,branch1]
    mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
    filt_exp_1x1 = conv2d(mixed,2048,1,1,pad,False,name=name+'fin1x1')
    final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=name+'act_saling')([x, filt_exp_1x1])
    return final_lay
def InceptionResNet2(input_shape, nclass):
    img_input = Input(shape=input_shape)

    x = conv2d(img_input,32,3,2,'valid',True,name='conv1')
    x = conv2d(x,32,3,1,'valid',True,name='conv2')
    x = conv2d(x,64,3,1,'valid',True,name='conv3')

    x_11 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_11'+'_maxpool_1')(x)
    x_12 = conv2d(x,64,3,1,'valid',True,name='stem_br_12')

    x = Concatenate(axis=3, name = 'stem_concat_1')([x_11,x_12])

    x_21 = conv2d(x,64,1,1,'same',True,name='stem_br_211')
    x_21 = conv2d(x_21,64,[1,7],1,'same',True,name='stem_br_212')
    x_21 = conv2d(x_21,64,[7,1],1,'same',True,name='stem_br_213')
    x_21 = conv2d(x_21,96,3,1,'valid',True,name='stem_br_214')

    x_22 = conv2d(x,64,1,1,'same',True,name='stem_br_221')
    x_22 = conv2d(x_22,96,3,1,'valid',True,name='stem_br_222')

    x = Concatenate(axis=3, name = 'stem_concat_2')([x_21,x_22])

    x_31 = conv2d(x,192,3,1,'valid',True,name='stem_br_31')
    x_32 = MaxPooling2D(3,strides=1,padding='valid',name='stem_br_32'+'_maxpool_2')(x)
    x = Concatenate(axis=3, name = 'stem_concat_3')([x_31,x_32])


    #Inception-ResNet-A modules
    x = incresA(x,0.15,name='incresA_1')
    x = incresA(x,0.15,name='incresA_2')
    x = incresA(x,0.15,name='incresA_3')
    x = incresA(x,0.15,name='incresA_4')

    #35 × 35 to 17 × 17 reduction module.
    x_red_11 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_1')(x)

    x_red_12 = conv2d(x,384,3,2,'valid',True,name='x_red1_c1')

    x_red_13 = conv2d(x,256,1,1,'same',True,name='x_red1_c2_1')
    x_red_13 = conv2d(x_red_13,256,3,1,'same',True,name='x_red1_c2_2')
    x_red_13 = conv2d(x_red_13,384,3,2,'valid',True,name='x_red1_c2_3')

    x = Concatenate(axis=3, name='red_concat_1')([x_red_11,x_red_12,x_red_13])

    #Inception-ResNet-B modules
    x = incresB(x,0.1,name='incresB_1')
    x = incresB(x,0.1,name='incresB_2')
    x = incresB(x,0.1,name='incresB_3')
    x = incresB(x,0.1,name='incresB_4')
    x = incresB(x,0.1,name='incresB_5')
    x = incresB(x,0.1,name='incresB_6')
    x = incresB(x,0.1,name='incresB_7')

    #17 × 17 to 8 × 8 reduction module.
    x_red_21 = MaxPooling2D(3,strides=2,padding='valid',name='red_maxpool_2')(x)

    x_red_22 = conv2d(x,256,1,1,'same',True,name='x_red2_c11')
    x_red_22 = conv2d(x_red_22,384,3,2,'valid',True,name='x_red2_c12')

    x_red_23 = conv2d(x,256,1,1,'same',True,name='x_red2_c21')
    x_red_23 = conv2d(x_red_23,256,3,2,'valid',True,name='x_red2_c22')

    x_red_24 = conv2d(x,256,1,1,'same',True,name='x_red2_c31')
    x_red_24 = conv2d(x_red_24,256,3,1,'same',True,name='x_red2_c32')
    x_red_24 = conv2d(x_red_24,256,3,2,'valid',True,name='x_red2_c33')

    x = Concatenate(axis=3, name='red_concat_2')([x_red_21,x_red_22,x_red_23,x_red_24])

    #Inception-ResNet-C modules
    x = incresC(x,0.2,name='incresC_1')
    x = incresC(x,0.2,name='incresC_2')
    x = incresC(x,0.2,name='incresC_3')

    #TOP
    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.6)(x)
    x = Dense(nclass, activation='softmax')(x)


    model = Model(img_input,x,name='inception_resnet_v2')

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


"""
TRAINING
==================================================================================
Change filepath accordingly
"""

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
filepath = 'weight' + filename_extension + '.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
                        verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor='val_loss', 
                      mode='min', 
                      patience=4, restore_best_weights=True)

callbacks_list = [checkpoint, early]


input_shape=(nrows,ncolumns,1)
nclass=24

"""
SELECT YOUR MODEL
"""
# model = from_paper(input_shape,nclass)
model = VGG19(input_shape,nclass)
#model = CaffeNet(input_shape,nclass)
# model = ResNeXt50(input_shape=input_shape, classes=nclass)
# model = InceptionResNet2(input_shape=input_shape, classes=nclass)

history = model.fit_generator(train_gen,
                              steps_per_epoch=nclass*1000/batch_size,          
                              validation_data=val_gen,
                              validation_steps=batch_size, 
                              epochs=40, verbose=1,
                              callbacks=callbacks_list)

model.save_model('model' + filename_extension + '.h5')


"""
VISUALISATION
================================================================================
"""
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

plt.savefig('accuracy_loss' + filename_extension + '.png')