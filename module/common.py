# coding: utf-8
# -----------------------------------------
# define common module
# -----------------------------------------
import tensorflow.keras.models as module
import tensorflow.keras.layers as nn
import tensorflow as tf
from tensorflow.keras import regularizers

from config.fcos_config import config as cfg

class conv_layer(module.Model):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 padding='same',
                 **kwargs):
        super(conv_layer, self).__init__(**kwargs)
        self.conv = nn.Conv2D(filters, kernel_size, strides, padding,
                              kernel_regularizer= regularizers.l2(cfg.weight_decay),
                              bias_regularizer=regularizers.l2(cfg.weight_decay),
                              name=self.name+'_conv')
        self.bn = nn.BatchNormalization(name=self.name+'_bn')
        self.relu = nn.ReLU(name=self.name+'_relu')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs,training=training)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x