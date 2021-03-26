from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Flatten, Dense, Permute
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
import numpy as np
import cv2

# -----------------------------#
#   计算原始输入图像
#   每一次缩放的比例
# -----------------------------#
def calculateScales(img):
    copy_img = img.copy()

    pr_scale = 1.0

    # 获取原始图像宽高
    h, w, _ = copy_img.shape

    # 计算调整后的图像宽高
    if min(w, h) > 500:
        pr_scale = 500.0 / min(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)
    elif max(w, h) < 500:
        pr_scale = 500.0 / max(h, w)
        w = int(w * pr_scale)
        h = int(h * pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h, w)

    # 计算缩放比例
    while minl >= 12:
        scales.append(pr_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    return scales

# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    cls_prob = np.swapaxes(cls_prob, 0, 1)  # 获取预测人脸置信度得分
    roi = np.swapaxes(roi, 0, 2)

    stride = 0
    # stride略等于2
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)

    # 获取置信度得分大于阈值的人脸框位置
    (x, y) = np.where(cls_prob >= threshold)

    boundingbox = np.array([x, y]).T
    # 找到对应原图的位置
    bb1 = np.fix((stride * (boundingbox) + 0) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    boundingbox = np.concatenate((bb1, bb2), axis=1)

    # 获取人脸框位置调整量
    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]

    # 将置信度得分和人脸框位置调整量转成ndarray
    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T

    # 获取人脸框在原始图像中位置
    boundingbox = boundingbox + offset * 12.0 * scale
    rectangles = np.concatenate((boundingbox, score), axis=1)

    # 转化成正方形
    rectangles = rect2square(rectangles)
    pick = []
    # 遍历人脸框，防止人脸框越界
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])

    # 对人脸框做非极大值抑制处理
    return NMS(pick, 0.3)

# -----------------------------#
#   将长方形调整为正方形
# -----------------------------#
def rect2square(rectangles):
    # 计算人脸框宽和高
    # print(rectangles.shape)

    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    # 计算宽高中的最大值
    l = np.maximum(w, h).T

    # 调整人脸框为正方形
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5

    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles

# -------------------------------------#
#   非极大抑制
# -------------------------------------#
def NMS(rectangles, threshold):
    # 如果无框则返回
    if len(rectangles) == 0:
        return rectangles
    # 将人脸框转ndarray
    boxes = np.array(rectangles)
    # 获取人脸框坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # 获取人脸框置信度
    s = boxes[:, 4]
    # 计算人脸框面积
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    # 对人脸框置信度得分排序
    I = np.array(s.argsort())
    pick = []
    while len(I) > 0:
        # 计算置信度得分最高的人脸框x1坐标与其他框的x1坐标最大值
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        # 计算置信度得分最高的人脸框y1坐标与其他框的y1坐标最大值
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        # 计算置信度得分最高的人脸框x2坐标与其他框的x2坐标最小值
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        # 计算置信度得分最高的人脸框y2坐标与其他框的y2坐标最小值
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])

        # 计算重合区域宽高
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        # 计算重合区域面积
        inter = w * h
        # 计算交并比IOU
        o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o <= threshold)[0]]
    result_rectangle = boxes[pick].tolist()

    return result_rectangle

# -------------------------------------#
#   对pnet处理后的结果进行处理
# -------------------------------------#
def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]  # 获取预测人脸置信度得分
    pick = np.where(prob >= threshold)  # 获取置信度得分大于阈值的索引

    # 人脸框转ndarray
    rectangles = np.array(rectangles)

    # 获取置信度大于阈值的人脸框
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    # 获取置信度得分大于阈值的置信度得分
    sc = np.array([prob[pick]]).T

    # 获取人脸框偏移量
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 计算人脸框宽高
    w = x2 - x1
    h = y2 - y1

    # 计算人脸框在原图上的位置
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T
    rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
    # 人脸框转成正方形
    rectangles = rect2square(rectangles)
    pick = []
    # 遍历人脸框，防止人脸框越界
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, sc])

    # 对人脸框做非极大值抑制处理
    return NMS(pick, 0.3)

# -------------------------------------#
#   对onet处理后的结果进行处理
# -------------------------------------#
def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
    prob = cls_prob[:, 1]  # 获取预测人脸置信度得分
    pick = np.where(prob >= threshold)  # 获取置信度得分大于阈值的索引
    rectangles = np.array(rectangles)   # 人脸框转ndarray

    # 获取置信度大于阈值的人脸框
    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    # 获取置信度得分大于阈值的置信度得分
    sc = np.array([prob[pick]]).T

    # 获取人脸框偏移量
    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    # 计算人脸框宽高
    w = x2 - x1
    h = y2 - y1

    # 计算人脸关键点在原图中的位置
    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T

    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T

    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T

    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T

    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    # 计算人脸框在原图上的位置
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),
                                axis=1)

    pick = []

    # 遍历人脸框，防止人脸框越界
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])

    # 对人脸框做非极大值抑制处理
    return NMS(pick, 0.3)

# -----------------------------#
#   Pnet网络粗略获取人脸框
#   输出bbox位置和是否有人脸
# -----------------------------#
def create_Pnet(weight_path):
    # 定义Pnet网络输入
    input = Input(shape=[None, None, 3])

    # 3x3卷积+prelu+最大池化
    # h,w,3 -> h/2,w/2,10
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 3x3卷积+prelu
    # h/2,w/2,10 -> h/2,w/2,16
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU2')(x)

    # 3x3卷积+prelu
    # h/2,w/2,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 1x1卷积，人脸分类
    # h/2, w/2, 2
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)

    # 1x1卷积，人脸框定位
    # h/2, w/2, 4
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])

    # 加载权重文件
    model.load_weights(weight_path, by_name=True)
    return model

# -----------------------------#
#   mtcnn的第二段Rnet
#   精修框
# -----------------------------#
def create_Rnet(weight_path):
    # 定义Rnet网络输入
    input = Input(shape=[24, 24, 3])

    # 3x3卷积+prelu+最大池化
    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 3x3卷积+prelu+最大池化
    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 3x3卷积+prelu
    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)

    # 特征拉直
    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)

    # 全连接层+prelu
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    # 全连接层+softmax人脸分类，全连接层人脸定位
    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

# -----------------------------#
#   mtcnn的第三段Onet
#   精修框并获得五个点
# -----------------------------#
def create_Onet(weight_path):
    # 定义Onet网络输入
    input = Input(shape=[48, 48, 3])

    # 3x3卷积+prelu+最大池化
    # 48,48,3 -> 23,23,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # 3x3卷积+prelu+最大池化
    # 23,23,32 -> 10,10,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 3x3卷积+prelu+最大池化
    # 8,8,64 -> 4,4,64
    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    # 3x3卷积+prelu
    # 4,4,64 -> 3,3,128
    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # 特征维度更换顺序
    # 3,3,128 -> 128,3,3
    x = Permute((3, 2, 1))(x)

    # 特征拉直+全连接层+prelu
    # 1152 -> 256
    x = Flatten()(x)
    x = Dense(256, name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 全连接+softmax人脸分类 256 -> 2
    classifier = Dense(2, activation='softmax', name='conv6-1')(x)
    # 全连接+人脸定位 256 -> 4
    bbox_regress = Dense(4, name='conv6-2')(x)
    # 全连接层回归人脸关键点 256 -> 10
    landmark_regress = Dense(10, name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')

    # 人脸检测函数
    def detectFace(self, img, threshold):
        # -----------------------------#
        #   归一化
        # -----------------------------#
        copy_img = (img.copy() - 127.5) / 127.5
        origin_h, origin_w, _ = copy_img.shape

        # -----------------------------#
        #   计算原始输入图像
        #   每一次缩放的比例
        # -----------------------------#
        scales = calculateScales(img)
        out = []
        # -----------------------------#
        #   粗略计算人脸框
        #   pnet部分
        # -----------------------------#
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(copy_img, (ws, hs))
            inputs = scale_img.reshape(1, *scale_img.shape)
            # Pnet预测人脸框
            ouput = self.Pnet.predict(inputs)
            out.append(ouput)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            # 有人脸的概率
            cls_prob = out[i][0][0][:, :, 1]
            # 其对应的框的位置
            roi = out[i][1][0]
            # 取出每个缩放后图片的长宽
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            # 解码过程
            rectangle = detect_face_12net(cls_prob, roi, out_side,
                                          1 / scales[i], origin_w, origin_h,threshold[0])
            rectangles.extend(rectangle)

        # 进行非极大抑制
        rectangles = NMS(rectangles, 0.8)

        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        # -----------------------------#
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        # Rnet预测人脸框
        out = self.Rnet.predict(predict_24_batch)
        # 置信度
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        # 如何调整某一张图片对应的rectangle
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        # 对pnet处理后的结果进行处理
        rectangles = filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles) == 0:
            return rectangles

        # -----------------------------#
        #   计算人脸框
        #   onet部分
        # -----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        # Onet预测人脸框和人脸关键点
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        # 对onet处理后的结果进行处理
        rectangles = filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        return rectangles

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
    # 绘制人脸框
    cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0))
    # 绘制关键点
    for i in range(5):
        cv2.circle(img,(int(points[i*2]),int(points[i*2+1])),4,(0,0,255),5)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result.jpg",img)