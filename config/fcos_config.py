from easydict import EasyDict as edit
import numpy as np

config = edit()
# config.CLASSES = ['crack', 'corrosion', 'breakage']
config.TRAIN_FILE = '../data/all_data.csv'
config.BATCH_SIZE = 1
config.MIN_SIZE = 800  # 训练
config.MAX_SIZE = 1024
config.STRIDES = [8, 16, 32, 64, 128]
# 根据实际box进行设置
config.MAX_SIZE_EACH_MAP = [0, 64, 128, 256, 512, np.inf]
config.MAX_OBJECTS = 100  #每张图不超过100个objects
config.label_def = ['_background','crack', 'corrosion', 'breakage']

# --------------------loss-----------------
config.ALPHA = 0.25
config.GAMMA = 2
config.loss_weight = [1, 1, 1] # balance classification loss, centerness loss, regression loss
config.SMOOTH = 1e-6  # avoid deviding by zero
# --------------------train----------------
config.lr_init = 1e-2
config.lr_end = 1e-6
config.weight_decay = 1e-4  # weight decay
config.momount = 0.9
config.weight_decay_step = [60000, 80000]
config.EPOCHS = 120  # 迭代轮次
config.model_saved_dir = 'E:/workspaces_e/fcos/models'
config.log_saved_dir = 'E:/workspaces_e/fcos/logs'
config.WRITE_IMAGE_PER_EPOCH = 20  # write image

# ----------- image process and show ---------------------
config.PIX_MEAN = [103.939, 116.779, 123.68]
config.COLOR_DEF = [[255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255]]

#----------------inference-----------------
config.CONFIDENCE = 0.1 # positive samples
