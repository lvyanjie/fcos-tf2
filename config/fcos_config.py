from easydict import EasyDict as edit
import numpy as np

config = edit()
# config.CLASSES = ['crack', 'corrosion', 'breakage']
config.TRAIN_FILE = '../data/all_data.csv'
config.BATCH_SIZE = 2
config.MIN_SIZE = 1024  # 训练
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

# --------------------train----------------
config.lr_init = 1e-2
config.lr_end = 1e-6
config.weight_decay = 1e-4  # weight decay
config.momount = 0.9
config.weight_decay_step = [60000, 80000]
config.EPOCHS = 120  # 迭代轮次
config.model_saved_dir = './data/model'

