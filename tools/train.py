#coding: utf-8
# ----------------------------------------
# 模型训练过程
# ----------------------------------------
import tensorflow as tf
from tensorflow import keras
import os

from model import BaseFcos
from Loss.loss import fcos_loss
from config.fcos_config import config as cfg
from core.dataset import Dataset

model_name = 'focs_0902'
model_saved = os.path.join(cfg.model_saved_dir, model_name)
if not os.path.exists(model_saved):
    os.makedirs(model_saved)

# 模型训练相关参数初始化
num_classes = len(cfg.label_def)# 1 is background
model = BaseFcos.FCOS(num_classes=num_classes)
print(model.summary())
trainset = Dataset()
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
steps_per_epoch = len(trainset)
total_steps = cfg.EPOCHS * steps_per_epoch
opt = keras.optimizers.SGD(lr=cfg.lr_init, momentum = cfg.momount)

# 观测值
avg_classification_loss = tf.keras.metrics.Mean(name='classification loss', dtype=tf.float32)
avg_centerness_loss     = tf.keras.metrics.Mean(name='centerness loss', dtype=tf.float32)
avg_giou_loss           = tf.keras.metrics.Mean(name='giou loss', dtype=tf.float32)
avg_total_loss          = tf.keras.metrics.Mean(name='total loss', dtype=tf.float32)

# 进度条打印
def process_bar(percent, start_str='', end_str='', total_length=20):
    bar = ''.join(['='] * int(percent * total_length)) + '>'
    bar = '\r' + start_str + '{:0>4.1f}% ['.format(percent * 100) + bar.ljust(total_length) + ']' + end_str
    print(bar, end='', flush=True)

def train_step(step, epoch, model, batch_image, batch_true, input_height, input_width):
    '''
    单次迭代训练过程
    :param step: for summary write
    :param epoch:
    :param model:
    :param image: 去均值以及padding之后的batch image
    :param bbox:  scale之后的bbox
    :param batch_valid_objects: 有效目标数量
    :return:
    '''

    with tf.GradientTape() as tape:
        batch_pred = model(batch_image)
        classification_loss, centerness_loss, giou_loss, total_loss =\
            fcos_loss(batch_true, batch_pred, num_classes, input_height, input_width)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        avg_classification_loss.update_state(classification_loss)
        avg_centerness_loss.update_state(centerness_loss)
        avg_giou_loss.update_state(giou_loss)
        avg_total_loss.update_state(total_loss)
        # epoch内通过进度条打印
        str_info = 'cls_loss:{:0>4.3f}, ' \
                   'center_loss:{:0>4.3f}, ' \
                    'giou_loss:{:0>4.3f}, '\
                    'total_loss:{:0>4.3f}'.\
            format(classification_loss,
                   centerness_loss,
                   giou_loss,
                   total_loss)
        process_bar(float(step) / steps_per_epoch, start_str='epoch{} '.format(epoch), end_str=str_info)

        global_steps.assign_add(1)

for epoch in range(cfg.EPOCHS):
    # train 过程
    for step, (batch_image, batch_bboxes, batch_valid_objects) in enumerate(trainset):
        batchsize, input_height, input_width, _ = batch_image.shape
        gt = BaseFcos.groundtruth(batch_image, batch_bboxes, batch_valid_objects)
        train_step(step, epoch, model, batch_image, gt, input_height, input_width)

    str_info = 'cls_loss:{:0>4.3f}, '\
                'center_loss:{:0>4.3f}, '\
                'giou_loss:{:0>4.3f}, '\
                'total_loss:{:0>4.3f}'.\
            format(avg_classification_loss.result(),
                   avg_centerness_loss.result(),
                   avg_giou_loss.result(),
                   avg_total_loss.result())
    process_bar(1.0, start_str='epoch{} '.format(epoch), end_str=str_info + '\n')

    avg_classification_loss.reset_states()
    avg_centerness_loss.reset_states()
    avg_giou_loss.reset_states()
    avg_total_loss.reset_states()

    model.save_weights('{}/fcos_model{}.h5'.format(model_saved, epoch))

