#coding:utf-8
# ------------------------------
# 每张图检测的目标数量不超过200个， 用于生成batch数据
# csv数据导入， csv数据格式： image_path, xmin, ymin, xmax, ymax, label
# ------------------------------
import numpy as np
import glob
import cv2
import json
import random
import pandas as pd

from config.fcos_config import config as cfg
from core import utils

from keras.applications.resnet50 import preprocess_input

class Dataset(object):
    def __init__(self):
        self.image_id, self.all_objects = self.load_data()
        self.batch_size = cfg.BATCH_SIZE
        self.max_objects = cfg.MAX_OBJECTS
        self.num_samples = len(self.image_id)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_data(self):
        df = pd.read_csv(cfg.TRAIN_FILE, header=0) # image_path, image_width, image_height, xmin, ymin, xmax, ymax, label
        image_id = list(df['image_path'])
        # shuffle
        random.shuffle(image_id)
        return image_id, df

    def shuffle_files(self):
        random.shuffle(self.image_id)

    def __iter__(self):
        return self

    def __next__(self):
        '''
        important
        :return: images。 labels
        '''
        batch_image = []
        # classification:
        batch_labels = np.zeros(shape=(self.batch_size, self.max_objects, 5)) # 4 bbox + 1 label
        batch_valid_count = np.zeros(shape=(self.batch_size,))

        num = 0
        if self.batch_count < self.num_batches:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index > self.num_samples:
                    break
                image_path = self.image_id[index]
                img, bbox, objects_number = self.get_data(image_path)
                # 对 image 进行预处理
                scale, scale_img = utils.resize_based_scale_new(img, cfg.MIN_SIZE, cfg.MAX_SIZE) # 获取scale
                # 根据scale，修正box

                scale_bbox = utils.map_bbox_to_scaleimage(bbox, scale)

                # 将数据处理成batch
                batch_image.append(scale_img)
                batch_labels[num, :objects_number, :] = scale_bbox
                batch_valid_count[num] = objects_number
                num+=1
            self.batch_count +=1
            #处理成batch image
            batch_image = self.get_batch_image(batch_image)
            return batch_image, batch_labels, batch_valid_count.astype(np.int32)
        else:
            self.batch_count=0
            self.shuffle_files()
            raise StopIteration

    def get_batch_image(self, batch_image):
        '''
        将获取的已经规约长度的图像处理成一个batch，用于模型喂入
        进行图像右部和下部零填充
        :param batch_image: list
        :return:
        '''
        # 获取 max height 和 max width
        batch_height = batch_width = cfg.MAX_SIZE

        final_batch_image = np.zeros(shape=(len(batch_image), batch_height, batch_width, 3))
        for i, image in enumerate(batch_image):
            height, width, _ = image.shape
            # 这种padding方式不需要对bbox的坐标进行transform
            final_batch_image[i,:height, :width, :] = image
        return final_batch_image

    def get_data(self, image_file):
        '''
        :param image_file:
        :param json_file:
        :return:
        '''
        img = cv2.imread(image_file) # initial image
        # 图像数据预处理
        # img = utils.process_input(img) # 去均值
        objects = self.all_objects[self.all_objects['image_path']==image_file]
        bboxes = []

        # initial_image
        for index, row in objects.iterrows():
            xmin, ymin, xmax, ymax, label_name = int(row['xmin']), \
                                                 int(row['ymin']), \
                                                 int(row['xmax']), \
                                                 int(row['ymax']), \
                                                 row['label']
            label = cfg.label_def.index(label_name) # str-number
            bboxes.append([xmin, ymin, xmax, ymax, label])
        bboxes = np.array(bboxes, dtype=np.float32)

        return img, bboxes, len(bboxes)

    def __len__(self):
        return self.num_batches

if __name__=='__main__':
    from model import BaseFcos
    import math
    import matplotlib.pyplot as plt

    dataset = Dataset()
    for i, (batch_image, batch_bboxes, batch_valid_objects) in enumerate(dataset):
        # 计算对应的groundthboxes
        batch_groundtruth = BaseFcos.groundtruth(batch_image, batch_bboxes, batch_valid_objects)
        classification = batch_groundtruth[:,:,:4] #前4列

        batch_size, input_height, input_width, _ = batch_image.shape

        label = np.argmax(classification, axis=-1)

        check_image = batch_image[0]
        check_gt = label[0]

        plt.imshow(check_image.astype(np.uint8))
        plt.show()

        points = []
        for stride in [8, 16, 32, 64, 128]:
            print(stride)
            f_height, f_width = math.ceil(input_height/stride), math.ceil(input_width/stride)
            all_points = f_height * f_width
            check_f = check_gt[:all_points]
            cehck_gt = check_gt[all_points:]
            s = np.reshape(check_f, (f_height, f_width))
            plt.imshow(s)
            plt.show()

