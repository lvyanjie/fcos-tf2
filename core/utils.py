# coding: utf-8
# --------------------------
# 常用工具集成
# --------------------------
import cv2
import numpy as np

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
    mean = [103.939, 116.779, 123.68]
    image[..., 0] -= mean[0]
    image[..., 1] -= mean[1]
    image[..., 2] -= mean[2]
    return image
