#coding: utf-8
#--------------------------------------
# FCOS tensorflow版本
#--------------------------------------

import tensorflow as tf
import tensorflow.keras.layers as nn
import tensorflow.keras.models as module
from module.fpn import retina_fpn
from module.common import conv_layer
from config.fcos_config import config as cfg

import numpy as np
import math

def decode_lrtb2box_tf(regression_result, input_height, input_width):
    '''
    '''
    # 将模型输出空间映射到[0, inf)
    regression_result = tf.exp(regression_result)
    decode_result = []
    f_counts = []
    for stride in cfg.STRIDES:
        f_h, f_w = tf.math.ceil(input_height / stride), tf.math.ceil(input_width / stride)
        f_h, f_w = tf.cast(f_h, tf.int32), tf.cast(f_w, tf.int32)
        f_count = f_h * f_w
        if len(f_counts) == 0:
            index_start = 0
        else:
            index_start = f_counts[-1]

        f_regression = regression_result[:, index_start:index_start + f_count, :]  # batchsize
        f_counts.append(f_count)

        pos_x = tf.range(0, f_w, 1)
        pos_y = tf.range(0, f_h, 1)
        f_pos = tf.meshgrid(pos_y, pos_x)

        f_pos = tf.convert_to_tensor(list(zip(tf.reshape(f_pos[0], [-1]), tf.reshape(f_pos[1], [-1]))),
                                     dtype=tf.float32)
        image_pos = tf.math.floor(stride / 2) + tf.multiply(f_pos, stride)
        image_pos_y, image_pos_x = image_pos[:, 0], image_pos[:, 1]
        batch_image_pos_y, batch_image_pos_x = tf.tile(image_pos_y, [cfg.BATCH_SIZE]), \
                                               tf.tile(image_pos_x, [cfg.BATCH_SIZE])
        batch_image_pos_y, batch_image_pos_x = tf.reshape(batch_image_pos_y, [cfg.BATCH_SIZE, f_count, 1]), \
                                               tf.reshape(batch_image_pos_x, [cfg.BATCH_SIZE, f_count, 1])
        l_pred, t_pred, r_pred, b_pred = tf.split(f_regression, num_or_size_splits=4, axis=-1)
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = batch_image_pos_x - l_pred, \
                                                     batch_image_pos_y - t_pred, \
                                                     batch_image_pos_x + r_pred, \
                                                     batch_image_pos_y + b_pred

        # 对value进行规约
        xmin_pred = tf.clip_by_value(xmin_pred, 0, input_width)
        ymin_pred = tf.clip_by_value(ymin_pred, 0, input_height)
        xmax_pred = tf.clip_by_value(xmax_pred, 0, input_width)
        ymax_pred = tf.clip_by_value(ymax_pred, 0, input_height)

        f_result = tf.concat([xmin_pred, ymin_pred, xmax_pred, ymax_pred], axis=-1)
        decode_result.append(f_result)

    #  batchsize, all_feature_count, 4
    decode_result = tf.concat(decode_result, axis=1)
    decode_result = tf.cast(decode_result, tf.float32)
    return decode_result

def decode_lrtb2box(regression_result, input_height, input_width, is_pred=False):
    '''
    decode regression result to box
    :param input_width: 用于计算feature size
    :param input_height: 同上
    :param regression_result: l_star, r_star, t_star, b_star
    :param is_gt: 是否时groundtruth，如果是predication，则需要进行exp操作
    :return:
    '''
    # 将regression_result映射到[0,inf)区间内
    if is_pred:
        regression_result = np.exp(regression_result)
    # 基于input_height, input_width 对每一层feature的regression结果进行抓取
    decode_result = []
    f_counts= []  #用于从总体结果中获取结果切片
    for stride in cfg.STRIDES:
        # feature map height and width
        f_h, f_w = math.ceil(input_height/stride), math.ceil(input_width/stride)
        f_count = int(f_h) * int(f_w)
        if len(f_counts)==0:
            index_start = 0
        else:
            index_start = f_counts[-1]   # 上次结束开始
        f_regression = regression_result[:, index_start:index_start + f_count,:] # batchsize
        f_counts.append(f_count)
        # feature pos
        pos_x = np.arange(0, f_w, 1)
        pos_y = np.arange(0, f_h, 1)
        f_pos = np.meshgrid(pos_y, pos_x)
        f_pos = np.array(list(zip(f_pos[0].flatten(), f_pos[1].flatten())))  # n, 2
        image_pos = math.floor(stride / 2) + f_pos * stride  # image position
        # image pos is
        image_pos_y, image_pos_x = image_pos[:,0], image_pos[:,1]  # image y, x

        # decode process
        # 对应每个feature map上的点映射的原图位置
        batch_image_pos_y, batch_image_pos_x = np.tile(image_pos_y, cfg.BATCH_SIZE),\
                                               np.tile(image_pos_x, cfg.BATCH_SIZE)

        batch_image_pos_y, batch_image_pos_x = np.reshape(batch_image_pos_y, (cfg.BATCH_SIZE, f_count, 1)),\
                                               np.reshape(batch_image_pos_x, (cfg.BATCH_SIZE, f_count, 1))
        l_pred, t_pred, r_pred, b_pred = np.split(f_regression, indices_or_sections=4, axis=-1)  # 最后有一个维度
        
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = batch_image_pos_x - l_pred,\
                                                     batch_image_pos_y - t_pred,\
                                                     batch_image_pos_x + r_pred,\
                                                     batch_image_pos_y + b_pred  # batchsize, n

        # 对value进行规约
        xmin_pred = np.clip(xmin_pred, a_min=0, a_max=input_width)
        ymin_pred = np.clip(ymin_pred, a_min=0, a_max=input_height)
        xmax_pred = np.clip(xmax_pred, a_min=0, a_max=input_width)
        ymax_pred = np.clip(ymax_pred, a_min=0, a_max=input_height)
        
        f_result = np.concatenate([xmin_pred, ymin_pred, xmax_pred, ymax_pred], axis=-1)  # 最后一维进行联结
        decode_result.append(f_result)

    #  batchsize, all_feature_count, 4
    decode_result = np.concatenate(decode_result, axis=1)
    return decode_result.astype(np.float32)

def gt_based_bbox_sigle_image(bbox, input_height, input_width, object_num):
    '''
    单图根据bbox计算ground truth
    :param bbox: objects bbox (xmin, ymin, xmax, ymax, label)
    :param object_num: 用于获取有效bbox
    :return:
    '''
    num_classes = len(cfg.label_def)
    strides = cfg.STRIDES
    valid_gtboxes = bbox[:object_num]  # gt_boxes

    all_gt = []
    for i, stride in enumerate(strides):
        # feature map
        feature_height, feature_width = math.ceil(input_height/stride), math.ceil(input_width/stride)
        # 基于feature_height, feature_width 进行grid网格
        gt_cls_labels = np.zeros(shape=(feature_height, feature_width)) #用于label定义
        gt_reg_targets = np.zeros(shape=(feature_height, feature_width, 4)) #用于regression traget定义
        gt_center_ness = np.zeros(shape=(feature_height, feature_width)) # 用于center ness
        gt_point_bbox = np.zeros(shape=(feature_height, feature_width, 4)) # 存放每个点对应的bbox，用于计算regression loss
        f_min_area = np.zeros(shape=(feature_height, feature_width)) #用于最佳box的映射

        feature_gt_boxes = valid_gtboxes.copy()
        feature_gt_boxes[:,:4] = feature_gt_boxes[:,:4] / stride  #bbox map映射
        # bbox叠加处理方式
        # 暂时没有想到更好的处理方法
        for j in range(feature_gt_boxes.shape[0]):
            # feature map上的点的坐标
            xmin, ymin, xmax, ymax, label = feature_gt_boxes[j]
            xmin_image, ymin_image, xmax_image, ymax_image, _ = valid_gtboxes[j] # 用于计算l_star, r_star, r_star, b_star
            xmin, ymin, xmax, ymax = int(math.ceil(xmin)), int(math.ceil(ymin)), int(math.floor(xmax)), int(math.floor(ymax))

            # box映射到当前feature map失败
            # 过小目标映射到当前feature map上消失， 即限制不等于0
            if xmax <= xmin or ymax <= ymin:
                continue

            label_region = gt_cls_labels[ymin:ymax, xmin:xmax]
            reg_region = gt_reg_targets[ymin:ymax, xmin:xmax, :] # regression
            center_region = gt_center_ness[ymin:ymax, xmin:xmax] # center ness
            area_region = f_min_area[ymin:ymax, xmin:xmax]  # 面积update
            point_box_region = gt_point_bbox[ymin:ymax, xmin:xmax, :] # gt point box

            # 计算 l_star, r_star, t_star, b_star
            pos_x = np.arange(xmin, xmax, 1)
            pos_y = np.arange(ymin, ymax, 1)
            pos_mid = np.meshgrid(pos_x, pos_y)
            pos = np.array(list(zip(pos_mid[0].flatten(), pos_mid[1].flatten())))
            pos_image = math.floor(stride / 2) + pos * stride  # 对应position映射到原图的位置
            x, y = pos_image[:, 0], pos_image[:, 1]
            l_star = x - xmin_image  # total_count, 1
            r_star = xmax_image - x  # total_count, 1
            t_star = y - ymin_image
            b_star = ymax_image - y

            # 生成对应网格的值
            l_star = np.reshape(l_star, [-1,1])
            t_star = np.reshape(t_star, [-1,1])
            r_star = np.reshape(r_star, [-1,1])
            b_star = np.reshape(b_star, [-1,1])
            grid_reg = np.concatenate([l_star, t_star, r_star, b_star], axis=-1)  # all_pos, 4
            grid_reg = np.reshape(grid_reg, ((ymax - ymin), (xmax - xmin), 4))

            # 计算 center-ness targets
            lr_star = np.concatenate([l_star, r_star], axis=-1)
            tb_star = np.concatenate([t_star, b_star], axis=-1)
            grid_center = np.sqrt((np.min(lr_star, axis=-1)/np.max(lr_star, axis=-1)) * (np.min(tb_star, axis=-1)/np.max(tb_star, axis=-1)))
            grid_center = np.reshape(grid_center, ((ymax - ymin), (xmax - xmin))) # the last dimentions

            # calculate area
            # value comapre, so the feature map is meaning
            bbox_area = (xmax - xmin) * (ymax - ymin)

            # 对于area为0的，表示还未进行初始化状态，则直接进行赋值操作
            y, x = np.where(area_region==0)   # 可能存在只有部分
            # 当y以及x不为空时，对相应位置的值进行更新
            if len(y)>0 and len(x)>0:
                label_region[y, x] = label
                reg_region[y, x, :] = grid_reg[y, x, :]
                center_region[y, x] = grid_center[y, x]
                area_region[y, x] = bbox_area
                point_box_region[y, x, :] = np.tile(np.array([[xmin_image, ymin_image, xmax_image, ymax_image]]), reps=[len(y), 1])

            # 对于原始矩阵中,area>0的部分，需要进行比较
            # a. label更新，只对面积大于当前bbox area的点进行更新, 更新的目标为当前的最佳obj（取面积小的object作为最佳映射目标）
            y, x = np.where(area_region > bbox_area)
            if len(y)>0 and len(x)>0:
                label_region[y, x] = label
                reg_region[y, x, :] = grid_reg[y, x, :]
                center_region[y, x] = grid_center[y, x]
                area_region[y, x] = bbox_area
                point_box_region[y, x, :] = np.tile(np.array([[xmin_image, ymin_image, xmax_image, ymax_image]]), reps=[len(y), 1])

            # 将update完成的region 更新到整图
            gt_cls_labels[ymin:ymax, xmin:xmax] = label_region
            f_min_area[ymin:ymax, xmin:xmax] = area_region # 面积update
            gt_reg_targets[ymin:ymax, xmin:xmax, :] = reg_region # regression
            gt_center_ness[ymin:ymax, xmin:xmax] = center_region # center ness
            gt_point_bbox[ymin:ymax, xmin:xmax, :] = point_box_region # point ground truth

        # 根据每个点的l_star, t_star, r_star, b_star 更新对应的class
        m_previous, m = cfg.MAX_SIZE_EACH_MAP[i], cfg.MAX_SIZE_EACH_MAP[i+1]
        max_gt_reg = np.max(gt_reg_targets, axis=-1)
        gt_cls_labels = np.where(((max_gt_reg<m_previous) | (max_gt_reg>m)), 0, gt_cls_labels)

        # 整合所有gt
        # 最后一个维度用于进行one-hot编码
        gt_cls_labels = np.eye(num_classes)[gt_cls_labels.astype(np.int32)]  #one-hot编码
        gt_center_ness = np.expand_dims(gt_center_ness, axis=-1)
        gt = np.concatenate([gt_cls_labels, gt_center_ness, gt_reg_targets, gt_point_bbox], axis=-1)
        gt = np.reshape(gt, newshape=(-1, num_classes + 1 + 4 + 4))
        all_gt.append(gt)

    all_gt = np.concatenate(all_gt, axis=0)
    # 每个点对应的 bbox 返回，方便loss计算
    return all_gt.astype(np.float32)

def groundtruth(batch_image, batch_bboxes, batch_valid_count):
    '''
    求取一个batch的groundtruth
    :param batch_image:
    :param batch_bboxes:
    :param batch_valid_count: 有效bbox的个数
    :return:
    '''
    batch_size, input_height, input_width, _ = batch_image.shape
    batch_groundtruth = []
    for i in range(batch_size):
        bboxes = batch_bboxes[i]
        gt = gt_based_bbox_sigle_image(bboxes, input_height, input_width, batch_valid_count[i])
        batch_groundtruth.append(gt)
    batch_groundtruth = np.array(batch_groundtruth, dtype=np.float32)
    return batch_groundtruth

# 构建head层
# 采用的是共享权重模式
class FcosHead(module.Model):
    '''
    all of the final activation fuction is "sigmoid"
    share weights
    '''
    def __init__(self, nun_classes, **kwargs):
        '''
        model unit
        :param nun_classes: output classes
        :param kwargs:
        '''
        super(FcosHead, self).__init__(**kwargs)
        self.conv1 = conv_layer(256, 3, 1, 'same', name='conv1')
        self.conv2 = conv_layer(256, 3, 1, 'same', name='conv2')
        self.conv3 = conv_layer(256, 3, 1, 'same', name='conv3')
        self.conv4 = conv_layer(256, 3, 1, 'same', name='conv4')
        # 论文中未强调 classification和regression不能共享权重
        self.classification = nn.Conv2D(nun_classes + 1, 1, 1, activation='sigmoid', name='classification') # num_classes: classification
                                                                                                            # 1: center-ness
        self.regression = nn.Conv2D(4, 1, 1, name='regression') #linear activation
        self.concat = nn.Concatenate(axis=-1) # stack all result

    @tf.function
    def call(self, inputs, training=None, mask=None):
        '''
        head层
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        cls_out = self.classification(x, training=training)
        reg_out = self.regression(x, training=training)
        result = self.concat([cls_out, reg_out]) # None, w, h, num_classes+1+4    all result stack
        # classification, regression
        return result

# 不共享权重模式的fcos_head
class classification(module.Model):
    def __init__(self, out_channel, **kwargs):
        '''
        fcos classification head
        :param num_classes:
        :param kwargs:
        '''
        super(classification, self).__init__(**kwargs)
        self.conv1 = conv_layer(256, 3, 1, 'same', name='conv1')
        self.conv2 = conv_layer(256, 3, 1, 'same', name='conv2')
        self.conv3 = conv_layer(256, 3, 1, 'same', name='conv3')
        self.conv4 = conv_layer(256, 3, 1, 'same', name='conv4')

        # 加1表示center_ness
        self.classification = nn.Conv2D(out_channel, 1, 1, activation='sigmoid', name='classification')

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        out = self.classification(x, training=training)
        return out

class regression(module.Model):
    def __init__(self, **kwargs):
        super(regression, self).__init__(**kwargs)
        self.conv1 = conv_layer(256, 3, 1, 'same', name='conv1')
        self.conv2 = conv_layer(256, 3, 1, 'same', name='conv2')
        self.conv3 = conv_layer(256, 3, 1, 'same', name='conv3')
        self.conv4 = conv_layer(256, 3, 1, 'same', name='conv4')

        self.regression = nn.Conv2D(4, 1, 1, activation=None, name='regression')

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)

        out = self.regression(x, training=training)
        return out

def get_backbone(backbone_name):
    if 'vovnet' in backbone_name:
        from backbone import vovnet as bk_model
    else:
        from backbone import vovnet as bk_model  # 后续加入其他backbone
    if backbone_name == 'vovnet27':
        return bk_model.VoVNet_27(including_top=False)
    if backbone_name == 'vovnet39':
        return bk_model.VoVNet_39(including_top=False)
    if backbone_name == 'vovnet57':
        return bk_model.VoVNet_57(including_top=False)

def FCOS(num_classes,
         pretrained_weights = None):
    '''
    函数式构建 model
    :param num_classes: 类别数量
    :param batch_size: 用于后面进行reshape
    :param pretrained_weights: load pretrained weights
    :return:
    '''
    inputs = nn.Input(shape=(None, None, 3))
    batch_size = tf.shape(inputs)[0]

    #all structure init
    backbone = get_backbone('vovnet27')
    backbone.build(input_shape=(None, None, None, 3))
    if pretrained_weights is not None:
        backbone.load_weights(pretrained_weights)
    fpn = retina_fpn(name='fpn')
    classification_head = classification(out_channel=num_classes + 1) # 1 is center_ness
    regression_head = regression()

    c_out, _ = backbone(inputs) #获取中间输出
    _, c3, c4, c5 = c_out
    # p3/8, p4/16, p5/32, p6/64, p7/128
    p_features = fpn([c3, c4, c5])

    # classfication an regression for all fpn features
    p_outs = [] # save all classification , center_ness, regression result

    for i, feature in enumerate(p_features):
        classification_out = classification_head(feature) # batch, feature_height, feature_width, num_classes+1
        regression_out = regression_head(feature)   #batch, feature_height, feature_width, 4
        # 对regression的结果进行解码


        out = tf.concat([classification_out, regression_out], axis=-1)

        out = tf.reshape(out, shape=[batch_size, -1, num_classes + 1 + 4]) # 将所有结果flatten到一个维度，用于结果连接，方便loss计算
        p_outs.append(out)
    p_outs = tf.concat(p_outs, axis=1) # 预测的所有 class，ness，以及box, 做了concat
    model = module.Model(inputs=inputs, outputs=p_outs, name='fcos')
    return model  # 只有在推理的时候进行nms

if __name__ == '__main__':
    model = FCOS(num_classes=4)
    model.build(input_shape=(2, 544, 544, 3))
    print(model.summary())

    # 查看训练参数
    for i, var in enumerate(model.trainable_weights):
        print(model.trainable_weights[i].name)

    print(model.output.shape) # batch_size, num, 7


