# coding: utf-8
# --------------------------------------
# vovnet
# --------------------------------------

import tensorflow.keras.models as module
import tensorflow.keras.layers as nn
import tensorflow as tf
from module.common import conv_layer


class stem(module.Model):
    def __init__(self,**kwargs):
        super(stem, self).__init__(**kwargs)
        self.conv1 = conv_layer(64, 3, 2, name='conv1')
        self.conv2 = conv_layer(64, 3, 1, name='conv2')
        self.conv3 = conv_layer(128, 3, 1, name='conv3')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x

class OSA_module(module.Model):
    def __init__(self, filters3x3, filters1x1, **kwargs):
        super(OSA_module, self).__init__(**kwargs)
        self.conv_layers = []

        for i in range(5):
            self.conv_layers.append(conv_layer(filters3x3, 3, 1, name='conv{}'.format(i+1)))

        self.concat = nn.Concatenate(axis=-1, name='concat')
        self.conv1x1 = conv_layer(filters1x1, 1, 1, name='conv1x1')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        cat_x = []
        x = inputs
        for i in range(5):
            x = self.conv_layers[i](x, training=training)
            cat_x.append(x)

        x = self.concat(cat_x)
        x = self.conv1x1(x, training=training)
        return x

class _VoVNet(module.Model):
    def __init__(self,
                 filters3x3,
                 filters1x1,
                 blocks,
                 including_top=False,
                 dense_unit = None,
                 **kwargs):
        '''
        base model
        :param filters13x3: 3x3 convolution filters, list
        :param filters1x1: 1x1 convolution filters, list
        :param blocks: OSA_module blocks in every block, list
        :param kwargs:
        '''
        super(_VoVNet, self).__init__(**kwargs)

        self.blocks = blocks
        self.including_top = including_top

        self.stem = stem(name='stage1_stem')
        self.stage_layers = []

        for i, block in enumerate(blocks):
            block_filters3x3 = filters3x3[i]
            block_filters1x1 = filters1x1[i]
            for j in range(block):
                layer_name = 'stage{}_osa_module{}'.format(i+1, j+1)
                self.stage_layers.append(OSA_module(block_filters3x3,
                                              block_filters1x1,
                                              name=layer_name))

        self.maxpool = nn.MaxPooling2D(pool_size=3, strides=2, padding='same')

        if including_top:
            assert dense_unit is not None
            self.global_average_pooling = nn.GlobalAveragePooling2D(name='glpool')
            self.flatten = nn.Flatten(name='flatten')
            self.Dense = nn.Dense(units=dense_unit, activation='softmax', name='prediction')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.stem(inputs, training=training)
        C_out = []

        for i, block in enumerate(self.blocks):
            for j in range(block):
                x = self.stage_layers[sum(self.blocks[:i]) + j](x, training=training)
            x = self.maxpool(x)
            C_out.append(x)  # 中间所有feature

        if self.including_top:
            x = self.global_average_pooling(x)
            x = self.flatten(x)
            x = self.Dense(x)

        return C_out, x

def VoVNet_27(including_top):
    return _VoVNet(filters3x3=[64, 80, 96, 112],
                   filters1x1=[128, 256, 384, 512],
                   blocks=[1, 1, 1, 1],
                   including_top=including_top,
                   dense_unit=1000)

def VoVNet_39(including_top):
    return _VoVNet(filters3x3=[128, 160, 192, 224],
                   filters1x1=[256, 512, 768, 1024],
                   blocks=[1, 1, 2, 2],
                   including_top=including_top,
                   dense_unit=1000)

def VoVNet_57(including_top):
    return _VoVNet(filters3x3=[128, 160, 192, 224],
                   filters1x1=[256, 512, 768, 1024],
                   blocks=[1, 1, 4, 3],
                   including_top=including_top,
                   dense_unit=1000)

if __name__=='__main__':
    model = VoVNet_57()
    model.build(input_shape=(None, 224, 224, 3))

    print(model.summary())

    for i, var in enumerate(model.trainable_weights):
        print(model.trainable_weights[i].name)

    test_numpy  = tf.random.uniform(shape=(1, 224, 224, 3))
    c_out, output = model(test_numpy)

    # 获取中间层输出
    c2, c3, c4, c5 = c_out

    print(c2.shape, c3.shape, c4.shape, c5.shape)


