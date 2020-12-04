# coding: utf-8
# ----------------------------------------
# define some tricks about model structure
# ----------------------------------------

import tensorflow as tf
import tensorflow.keras.layers as nn
import tensorflow.keras.models as module
from module.common import conv_layer

class retina_fpn(module.Model):
    def __init__(self, **kwargs):
        super(retina_fpn, self).__init__(**kwargs)
        self.conv5 = conv_layer(256, 1, 1, name='P5_conv')
        self.up_pool = nn.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.conv4 = conv_layer(256, 1, 1, name='P4_conv')
        self.conv3 = conv_layer(256, 1, 1, name='P3_conv')
        self.conv6 = conv_layer(256, 3, 2, padding='same', name='P6_conv')
        self.conv7 = conv_layer(256, 3, 2, padding='same', name='P7_conv')
        self.add = nn.Add()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        '''
        forward
        :param inputs: list   [C3, C4, C5]
        :param training:
        :param mask:
        :return:
        '''
        c3, c4, c5 = inputs
        #p5
        p5 = self.conv5(c5, training=training)
        #p4
        mid_p4 = self.up_pool(p5)
        p4 = self.add([self.conv4(c4), mid_p4])
        #p3
        mid_p3 = self.up_pool(p4)
        p3 = self.add([self.conv3(c3), mid_p3])
        #p6
        p6 = self.conv6(p5)
        #p7
        p7 = self.conv7(p6)

        return [p3, p4, p5, p6, p7]





