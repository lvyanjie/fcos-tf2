# coding: utf-8
# --------------------------
# loss
# --------------------------
import tensorflow as tf
from tensorflow import keras

from config.fcos_config import config as cfg
from model import BaseFcos

def fcos_loss(y_true, y_pred, num_classes, input_height, input_width):
    '''

    :param y_true: batchsize, n, num_classes + 1 + 4
    :param y_pred: batchsize, n, num_classes + 1 + 4
    :param num_classes: 类别数，用于进行label的切片
    :return:
    '''
    y_true_classification = y_true[:,:,:num_classes]
    y_true_centerness = y_true[:,:,num_classes]
    y_true_regression = y_true[:,:,-4:]   # true box

    y_pred_classification = y_pred[:,:,:num_classes]
    y_pred_centerness = y_pred[:,:,num_classes]
    y_pred_regression = y_pred[:,:,num_classes+1:]

    # only compute positive loss
    # shape: batchsize, numpoints
    label = tf.argmax(y_true_classification, axis=-1) # 获取对应的最大值索引，为0表示当前是negative

    y_true_centerness_valid = tf.gather_nd(y_true_centerness, tf.where(label>0))  # only compute label>0
    y_pred_centerness_valid = tf.gather_nd(y_pred_centerness, tf.where(label>0)) # 1D

    centerness_loss = center_ness_loss(y_true_centerness_valid, y_pred_centerness_valid)

    # regression loss
    # true regression decode
    # true_decode_bboxes = BaseFcos.decode_lrtb2box(y_true_regression, input_height, input_width)
    pred_decode_bboxes = BaseFcos.decode_lrtb2box_tf(y_pred_regression, input_height, input_width)

    # label shape: batchsize, num_points
    label_ref = tf.tile(tf.expand_dims(label, axis=-1), [1,1,4]) # batchsize, num_points, 4

    true_decode_bboxes_valid = tf.gather_nd(y_true_regression, tf.where(label_ref>0))  #1D
    pred_decode_bboxes_valid = tf.gather_nd(pred_decode_bboxes, tf.where(label_ref>0))
    true_decode_bboxes_valid = tf.reshape(true_decode_bboxes_valid, [-1, 4])
    pred_decode_bboxes_valid = tf.reshape(pred_decode_bboxes_valid, [-1, 4])

    tx1, ty1, tx2, ty2 = tf.split(true_decode_bboxes_valid, num_or_size_splits=4, axis=-1)  # num_points, 1
    px1, py1, px2, py2 = tf.split(pred_decode_bboxes_valid, num_or_size_splits=4, axis=-1)

    giou_loss = regression_loss([tx1, ty1, tx2, ty2], [px1, py1, px2, py2])

    # classification loss
    y_true_classification_valid = tf.gather_nd(y_true_classification, tf.where(label_ref>0))
    y_pred_classification_valid = tf.gather_nd(y_pred_classification, tf.where(label_ref>0))
    classification_loss = focal_loss(y_true_classification_valid,\
                                     y_pred_classification_valid,\
                                     alpha=cfg.ALPHA, gamma=cfg.GAMMA)

    weight_cls, weight_centerness, weight_reg = cfg.loss_weight

    total_loss = weight_cls * classification_loss + \
                 weight_centerness * centerness_loss + \
                 weight_reg * giou_loss
    return classification_loss, centerness_loss, giou_loss, total_loss

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    '''
    :param y_true: [batchsize, n, num_classes]  num_classes sigmoid activation
    :param y_pred:
    :return:
    '''
    epison = 1e-07

    # reshape
    # y_true = tf.reshape(y_true, [-1])  # flatten
    # y_pred = tf.reshape(y_pred, [-1])
    y_pred = tf.clip_by_value(y_pred, clip_value_min=epison, clip_value_max=1 - epison)
    p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    # alpha
    alpha_factor = tf.ones_like(y_true) * alpha
    alpha_t = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    # weight
    weight = alpha_t * tf.pow((1 - p_t), gamma)
    loss = - weight * tf.math.log(p_t)
    return tf.reduce_mean(loss)

def regression_loss(y_true, y_pred, smooth = 1e-6):
    '''

    :param y_true: [xmin, ymin, xmax, ymax]
    :param y_pred: [pred_xmin, pred_ymin, pred_xmax, pred_ymax]
    :param smooth: 平滑系数，避免分母为0
    :return:
    '''
    gx1, gy1, gx2, gy2 = y_true  # num_points, 1
    px1, py1, px2, py2 = y_pred

    garea = (gx2 - gx1) * (gy2 - gy1)  # num_points, 1
    parea = (px2 - px1) * (py2 - py1)

    inter_x1, inter_y1, inter_x2, inter_y2 = tf.maximum(gx1, px1),\
                                             tf.maximum(gy1, py1),\
                                             tf.minimum(gx2, px2),\
                                             tf.minimum(gy2, py2)
    inter_w, inter_h = inter_x2 - inter_x1, inter_y2 - inter_y1
    # 规约负值
    inter_w = tf.where(inter_w<0, 0, inter_w)
    inter_h = tf.where(inter_h<0, 0, inter_h)
    inter_area = inter_w * inter_h

    contain_x1, contain_y1, contain_x2, contain_y2 = tf.minimum(gx1, px1),\
                                             tf.minimum(gy1, py1),\
                                             tf.maximum(gx2, px2),\
                                             tf.maximum(gy2, py2)
    contain_w, contain_h = contain_x2 - contain_x1, contain_y2 - contain_y1
    contain_area = contain_w * contain_h

    union_area = garea + parea - inter_area
    iou = (inter_area + smooth) / (union_area + smooth)
    giou = iou - ((contain_area - union_area + smooth) / (contain_area + smooth))
    giou_loss = 1 - giou

    return tf.cast(tf.reduce_mean(giou_loss), tf.float32)

def center_ness_loss(y_true, y_pred):
    '''
    只针对class > 0 的 point计算 center_ness_loss
    :param y_true:  1D Dimentions
    :param y_pred:
    :return:
    '''
    # loss
    loss = keras.losses.binary_crossentropy(y_true=y_true,
                                            y_pred=y_pred)  # from_logits=False, label_smoothing
    loss = tf.reduce_mean(loss)
    return loss
