# coding: utf-8
# --------------------------
# 常用工具集成
# --------------------------
import cv2
import numpy as np
from config.fcos_config import config as cfg


def resize_based_scale_new(image, min_size, max_size):
    '''
    修改图像缩放函数
    最短是min_size, 最长小于等于 max_size
    '''
    img_height, img_width, _ = image.shape
    min_image_size, max_image_size = min(img_height, img_width), max(img_height, img_width)

    # 短边
    scale = min_size / min_image_size

    # 长边resize尺寸
    max_image_size_scale = max_image_size * scale
    if max_image_size_scale > max_size:
        scale = max_size / max_image_size

    resize_image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    return scale, resize_image

def resize_based_scale(image, min_size, max_size):
    '''
    # 对图像数据同比例缩放
    :param image:
    :param min_size:
    :param max_size:
    :return:
    '''
    img_height, img_width, _ = image.shape
    # 优先解决
    min_length = min(img_height, img_width)
    scale = min_size / min_length

    # 基于当前scale，求取height和width
    resize_height = img_height * scale
    resize_width = img_width * scale

    if resize_height > max_size or resize_width > max_size:
        max_length = max(img_height, img_width)
        scale = max_size / max_length

    resize_image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

    return scale, resize_image

def map_bbox_to_scaleimage(bbox, scale):
    '''
    基于scale，对bbox进行映射
    :param bbox:
    :param scale:
    :return:
    '''
    bbox_xy = bbox[:,:4] #xmin, ymin, xmax, ymax
    scale_bbox = bbox_xy * scale
    bbox[:,:4] = scale_bbox
    return bbox

def process_input(image):
    '''
    # 对图像数据进行预处理
    :param batch_image:
    :return:
    '''
    # imagenet dataset mean
    # B, G, R
    image = image.astype(np.float32)
    image[..., 0] -= cfg.PIX_MEAN[0]
    image[..., 1] -= cfg.PIX_MEAN[1]
    image[..., 2] -= cfg.PIX_MEAN[2]
    return image

# def plot_image_pred_batch(batch_image, batch_pred_result):
#     '''
#
#     :param batch_image:
#     :param batch_pred_result: numclasses
#     :return:
#     '''
#     num_clases = len(cfg.label_def)
#     plot_batch_result = batch_image.copy()
#     for i in range(batch_image.shape[0]):
#         image_initial = batch_image[i]
#         image_initial[..., 0] += cfg.PIX_MEAN[0]
#         image_initial[..., 1] += cfg.PIX_MEAN[1]
#         image_initial[..., 2] += cfg.PIX_MEAN[2]
#
#         pred_result = batch_pred_result[i]
#         classification_pred = pred_result[:, :]
#
#         for box in bboxes:
#             xmin, ymin, xmax, ymax, label = box
#             cv2.rectangle(image_initial, [xmin, ymin], [xmax, ymax], color=cfg.COLOR_DEF[label - 1], thickness=2) # for index
#             cv2.putText(image_initial, cfg.label_def[label], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, cfg.COLOR_DEF[label-1], 2)
#
#     return image_initial