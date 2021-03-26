import cv2
import numpy as np
from mtcnn import *
import math

#-------------------------------------#
#   人脸对齐
#-------------------------------------#
def Alignment(img,landmark):
    # 计算两点间x方向和y方向的差
    x = landmark[0,0] - landmark[1,0]
    y = landmark[0,1] - landmark[1,1]

    if x==0:
        angle = 0
    else:
        # 计算旋转角度
        angle = math.atan(y/x)*180/math.pi
    # 获取图像中心
    center = (img.shape[1]//2, img.shape[0]//2)

    # 计算旋转矩阵
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 旋转变换
    new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    # 遍历所有关键点，对所有关键点进行同样的旋转变换
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0,0]*landmark[i,0]+RotationMatrix[0,1]*landmark[i,1]+RotationMatrix[0,2])
        pts.append(RotationMatrix[1,0]*landmark[i,0]+RotationMatrix[1,1]*landmark[i,1]+RotationMatrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


if __name__ == "__main__":
    mtcnn_model = mtcnn()
    img = cv2.imread("./image/timg.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 检测人脸
    threshold = [0.5, 0.8, 0.9]
    rectangles = mtcnn_model.detectFace(img, threshold)

    # 转化成正方形
    rectangles = rect2square(np.array(rectangles))
    rectangle = rectangles[0]
    bbox = rectangle[0:4]
    points = rectangle[-10:]

    # 绘制关键点
    for i in range(5):
        cv2.circle(img,(int(points[i*2]),int(points[i*2+1])),4,(0,0,255),5)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 记下他们的landmark
    landmark = (np.reshape(rectangle[5:15], (5, 2)) - np.array([int(rectangle[0]), int(rectangle[1])])) / (
            rectangle[3] - rectangle[1])
    # 裁剪人脸图像
    crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]

    # 对其人脸
    new_img, _ = Alignment(crop_img, landmark)

    cv2.imwrite("image/crop_img.jpg",crop_img)
    cv2.imwrite("image/alignment_img.jpg", new_img)
