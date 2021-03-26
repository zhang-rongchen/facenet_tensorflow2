import os
from mtcnn import *
from model import build_model
from faceAlignment import Alignment

from sklearn.metrics import pairwise_distances

#---------------------------------#
#   图片预处理
#   高斯归一化
#---------------------------------#
def pre_process(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y
#---------------------------------#
#   l2标准化
#---------------------------------#
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output
#---------------------------------#

# 定义facenet人脸检测处理类
class Face_Rec():
    def __init__(self):
        # 创建mtcnn对象
        # 检测图片中的人脸
        self.mtcnn_model = mtcnn()
        # 门限函数
        self.threshold = [0.5, 0.7, 0.7]

        # 载入facenet
        # 将检测到的人脸转化为128维的向量
        self.facenet_model, self.image_embedder = build_model()
        # model.summary()
        model_path = './weights/model.20-0.0142.h5'
        # 加载模型文件
        self.facenet_model.load_weights(model_path)

    def recognize(self, rec_image_path,database_path,threshold = 0.3):

        rec_image = cv2.imread(rec_image_path)
        rec_image = cv2.cvtColor(rec_image, cv2.COLOR_BGR2RGB)

        # 检测人脸
        rec_rectangles = self.mtcnn_model.detectFace(rec_image, self.threshold)
        if len(rec_rectangles) == 0:
            return None
        # 转化成正方形
        rec_rectangles = rect2square(np.array(rec_rectangles))
        # facenet要传入一个160x160的图片
        rec_rectangles = rec_rectangles[0]
        # 记下landmark
        rec_landmark = (np.reshape(rec_rectangles[5:15], (5, 2)) - np.array(
            [int(rec_rectangles[0]), int(rec_rectangles[1])])) / (
                               rec_rectangles[3] - rec_rectangles[1]) * 160

        # 裁剪人脸图像
        rec_crop_img = rec_image[int(rec_rectangles[1]):int(rec_rectangles[3]),
                       int(rec_rectangles[0]):int(rec_rectangles[2])]
        rec_crop_img = cv2.resize(rec_crop_img, (160, 160))

        # 对齐人脸
        rec_new_img, _ = Alignment(rec_crop_img, rec_landmark)
        rec_new_img = np.expand_dims(rec_new_img, 0)
        rec_face_img = pre_process(rec_new_img)
        rec_pre = self.facenet_model.predict([rec_face_img, rec_face_img, rec_face_img])
        rec_pre = rec_pre[:, 0:128]
        rec_pre = l2_normalize(np.concatenate(rec_pre))
        rec_feature = np.reshape(rec_pre, [128])

        distances = []
        labels = []

        for root, dirs, files in os.walk(database_path):
            for file in files:
                database_path = os.path.join(root, file)

                # 读取数据库中的图像,并提取特征
                database_image = cv2.imread(database_path)

                # 图像标准化，为了提取特征
                database_image = cv2.cvtColor(database_image, cv2.COLOR_BGR2RGB)

                # 检测人脸
                database_rectangles = self.mtcnn_model.detectFace(database_image, self.threshold)
                if len(database_rectangles) ==0:
                    continue
                # 转化成正方形
                database_rectangles = rect2square(np.array(database_rectangles))

                database_rectangles = database_rectangles[0]
                # 记下landmark
                database_landmark = (np.reshape(database_rectangles[5:15], (5, 2)) - np.array([int(database_rectangles[0]), int(database_rectangles[1])])) / (
                        database_rectangles[3] - database_rectangles[1]) * 160

                # 裁剪人脸图像
                database_crop_img = database_image[int(database_rectangles[1]):int(database_rectangles[3]), int(database_rectangles[0]):int(database_rectangles[2])]
                H,W =  database_crop_img.shape[:2]
                if H <= 0 or W <=0:
                    continue

                labels.append()

                database_crop_img = cv2.resize(database_crop_img, (160, 160))

                # 对齐人脸
                database_new_img, _ = Alignment(database_crop_img, database_landmark)
                #cv2.imwrite("face/"+leftImageList[i].split("/")[-1],database_new_img)
                # 扩展一个维度
                database_new_img = np.expand_dims(database_new_img, 0)
                # 预处理
                database_face_img = pre_process(database_new_img)
                # 模型预测得到特征
                database_pre = self.facenet_model.predict([database_face_img, database_face_img, database_face_img])
                database_pre = database_pre[:, 0:128]
                # 正则化处理
                database_pre = l2_normalize(np.concatenate(database_pre))
                database_feature = np.reshape(database_pre, [128])

                distance = self.calculate_distance(rec_feature,database_feature)
                distances.append(distance)
                labels.append(file)
        distance_norm = []
        for i in range(len(distances)):
            distance_norm[i] = (distances[i] - np.min(distances)) / (np.max(distances) - np.min(distances))
        distance_norm = np.asarray(distance_norm)
        min_index = np.argmin(distance_norm)
        if distance_norm[min_index] < threshold:
            return labels[min_index]
        else:
            return None


    def calculate_distance(self,rec_feature,database_feature):
        dis = 1 - pairwise_distances(rec_feature, database_feature, metric='cosine')
        return dis

if __name__ == "__main__":
    face_rec = Face_Rec()
    rec_image_path = "rec_image.jpg"
    database_path = "image"
    # 识别
    label = face_rec.recognize(rec_image_path,database_path)
    print(label)



