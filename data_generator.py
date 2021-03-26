# encoding=utf-8
import json
import os
import cv2
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.utils import Sequence
from config import batch_size, img_size, channel, embedding_size

from mtcnn import *
from faceAlignment import Alignment

# 定义数据生成器
class DataGenSequence(Sequence):
    def __init__(self,image_folder):
        print('loading train samples')
        self.image_folder = image_folder
        # 打开文件夹
        with open('data/lfw_val_triplets.json', 'r') as file:
            self.samples = json.load(file)

        # 创建MTCNN
        self.mtcnn_model = mtcnn()
        # 人脸检测阈值
        self.threshold = [0.5, 0.8, 0.8]

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size   # 计算当前batch数据索引起点

        # 计算当前batch大小
        length = min(batch_size, (len(self.samples) - i))
        # 声明一个batch数据
        batch_inputs = np.empty((3, length, img_size, img_size, channel), dtype=np.float32)
        # 声明label数据
        batch_dummy_target = np.zeros((length* 3, embedding_size), dtype=np.float32)

        # 遍历获取一个batch数据
        for i_batch in range(length):
            sample = self.samples[i + i_batch]
            # print(sample)
            for j, role in enumerate(['a', 'p', 'n']):
                image_name = sample[role]  # 获取图像名
                filename = os.path.join(self.image_folder, image_name)
                img = cv2.imread(filename)  # 读取图像数据
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   # 转成RGB

                # 人脸检测
                rectangles = self.mtcnn_model.detectFace(img, self.threshold)

                num_faces = len(rectangles)
                if num_faces > 0:
                    # 转化成正方形
                    rectangles = rect2square(np.array(rectangles))
                    rectangle = rectangles[0]
                    # 记下他们的landmark
                    landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array(
                        [int(rectangle[0]), int(rectangle[1])])) / (
                                       rectangle[3] - rectangle[1]) * 160
                    # 裁剪人脸图像
                    crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

                    if crop_img.shape[0] > 0 and crop_img.shape[1] > 1:
                        crop_img = cv2.resize(crop_img, (160, 160))
                        # 对齐人脸
                        image, _ = Alignment(crop_img, landmark)
                    else:
                        image = cv2.resize(img, (img_size, img_size), cv2.INTER_CUBIC)

                else:
                    image = cv2.resize(img, (img_size, img_size), cv2.INTER_CUBIC)

                # 输入图像预处理
                batch_inputs[j, i_batch] = preprocess_input(image)
        return np.vstack((batch_inputs[0], batch_inputs[1], batch_inputs[2])), batch_dummy_target

    def on_epoch_end(self):
        np.random.shuffle(self.samples)

def revert_pre_process(x):
    return ((x + 1) * 127.5).astype(np.uint8)


if __name__ == '__main__':
    data_gen = DataGenSequence("lfw")
