import os
from mtcnn import *
from model import build_model
from faceAlignment import Alignment
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics import roc_curve
import joblib
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
        model_path = 'weights/model.29-0.0023.h5'
        # 加载模型文件
        self.facenet_model.load_weights(model_path)

        # 读取文件
        self.pairs = self.read_pairs("data/pairs.txt")
        # 获取图像对和标记：0-不同人，1-同一个人
        self.lefts,self.rights,self.labels = self.get_paths(self.pairs)

        # 提取特征
        self.leftfeatures, self.rightfeatures = self.extractFeature(self.lefts,self.rights)


    def extractFeature(self, leftImageList, rightImageList):

        leftfeatures = []
        rightfeatures= []
        labels = []
        for i in range(0, len(leftImageList)):

            if (i % 200 == 0):
                print("there are %d images done!" % i)

            # 读取左边图像,并提取特征
            imagel = cv2.imread(leftImageList[i])
            print(i,"   ",leftImageList[i])
            # 图像标准化，为了提取特征
            imagel = cv2.cvtColor(imagel, cv2.COLOR_BGR2RGB)

            # 检测人脸
            rectanglesl = self.mtcnn_model.detectFace(imagel, self.threshold)
            if len(rectanglesl) ==0:
                continue
            # 转化成正方形
            rectanglesl = rect2square(np.array(rectanglesl))

            rectanglel = rectanglesl[0]
            # 记下landmark
            landmarkl = (np.reshape(rectanglel[5:15], (5, 2)) - np.array([int(rectanglel[0]), int(rectanglel[1])])) / (
                    rectanglel[3] - rectanglel[1]) * 160

            # 裁剪人脸图像
            crop_imgl = imagel[int(rectanglel[1]):int(rectanglel[3]), int(rectanglel[0]):int(rectanglel[2])]
            H,W =  crop_imgl.shape[:2]
            if H <= 0 or W <=0:
                continue
            crop_imgl = cv2.resize(crop_imgl, (160, 160))

            # 对齐人脸
            new_imgl, _ = Alignment(crop_imgl, landmarkl)
            cv2.imwrite("face/"+leftImageList[i].split("/")[-1],new_imgl)
            # 扩展一个维度
            new_imgl = np.expand_dims(new_imgl, 0)
            # 预处理
            face_imgl = pre_process(new_imgl)
            # 模型预测得到特征
            prel = self.facenet_model.predict([face_imgl, face_imgl, face_imgl])
            prel = prel[:, 0:128]
            # 正则化处理
            prel = l2_normalize(np.concatenate(prel))
            fl = np.reshape(prel, [128])

            # 读取右边图像,并提取特征
            imager = cv2.imread(rightImageList[i])
            # 图像标准化，为了提取特征
            imager = cv2.cvtColor(imager, cv2.COLOR_BGR2RGB)

            # 检测人脸
            rectanglesr = self.mtcnn_model.detectFace(imager, self.threshold)
            if len(rectanglesr) == 0:
                continue
            # 转化成正方形
            rectanglesr = rect2square(np.array(rectanglesr))
            # facenet要传入一个160x160的图片
            rectangler = rectanglesr[0]
            # 记下landmark
            landmarkr = (np.reshape(rectangler[5:15], (5, 2)) - np.array([int(rectangler[0]), int(rectangler[1])])) / (
                    rectangler[3] - rectangler[1]) * 160

            # 裁剪人脸图像
            crop_imgr = imager[int(rectangler[1]):int(rectangler[3]), int(rectangler[0]):int(rectangler[2])]

            H, W = crop_imgr.shape[:2]
            if H<=0 or W<=0:
                continue

            crop_imgr = cv2.resize(crop_imgr, (160, 160))

            # 对齐人脸
            new_imgr, _ = Alignment(crop_imgr, landmarkr)
            new_imgr = np.expand_dims(new_imgr, 0)
            face_imgr = pre_process(new_imgr)
            prer = self.facenet_model.predict([face_imgr, face_imgr, face_imgr])
            prer = prer[:, 0:128]
            prer = l2_normalize(np.concatenate(prer))
            fr = np.reshape(prer, [128])

            leftfeatures.append(fl)
            rightfeatures.append(fr)
            labels.append(self.labels[i])

        self.labels = labels
        return leftfeatures, rightfeatures

    def read_pairs(self, pairs_filename):
        pairs = []
        # 打开文件
        f = open(pairs_filename, 'r')
        while True:
            # 读取一行，并以空格分割
            line = f.readline().strip('\n').split()
            if not line:
                break
            if len(line) == 3 or len(line) == 4:
                pairs.append(line)
        return pairs

    # 获取测试图像路径和标记
    def get_paths(self, pairs):
        # 测试图像路径
        ori_path = 'face_dataset/'
        lefts = []
        rights = []
        labels = []

        for i in range(0, len(pairs)):
            if len(pairs[i]) == 3:
                left = ori_path + pairs[i][0] + '/' + pairs[i][0] + '_' + \
                       '%04d' % int(pairs[i][1]) + '.jpg'
                right = ori_path + '/' + pairs[i][0] + '/' + pairs[i][0] + '_' \
                        + '%04d' % int(pairs[i][2]) + '.jpg'
                label = 1
                lefts.append(left)
                rights.append(right)
                labels.append(label)
                print("****************lefts*******************", left)
                print("****************rights*******************", right)
                print("****************labels*******************", label)
            elif len(pairs[i]) == 4:
                left = ori_path + pairs[i][0] + '/' + pairs[i][0] + '_' + \
                       '%04d' % int(pairs[i][1]) + '.jpg'
                right = ori_path + '/' + pairs[i][2] + '/' + pairs[i][2] + '_' + \
                        '%04d' % int(pairs[i][3]) + '.jpg'
                label = 0
                print("****************lefts*******************", left)
                print("****************rights*******************", right)
                print("****************labels*******************", label)

            else:
                print("error!!!!")

        return lefts,rights,labels


    def calculate_accuracy(self, distance, labels, num):
        accuracy = {}
        predict = np.empty((num,))
        threshold = 0.1
        while threshold <= 0.98:
            for i in range(num):
                if distance[i] >= threshold:
                    predict[i] = 1
                else:
                    predict[i] = 0
            predict_right = 0.0
            for i in range(num):
                if predict[i] == labels[i]:
                    predict_right += 1.0
            current_accuracy = (predict_right / num)
            accuracy[str(threshold)] = current_accuracy
            threshold = threshold + 0.001
        # 将字典按照value排序
        temp = sorted(accuracy.items(), key=lambda d: d[1], reverse=True)
        highestAccuracy = temp[0][1]
        thres = temp[0][0]
        return highestAccuracy, thres

    # 绘制roc曲线函数
    def draw_roc_curve(self, fpr, tpr, title='cosine', save_name='roc_lfw.png'):
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic using: ' + title)
        plt.legend(loc="lower right")
        pathplt = save_name
        plt.savefig(pathplt)
        plt.show()

    def calculate_distance(self,leftfeatures,rightfeatures,labels):
        dis = 1 - pairwise_distances(leftfeatures, rightfeatures, metric='cosine')
        distance = np.empty((len(labels),))
        for i in range(len(labels)):
            distance[i] = dis[i][i]

        distance_norm = np.empty((len(labels)))
        for i in range(len(labels)):
            distance_norm[i] = (distance[i] - np.min(distance)) / (np.max(distance) - np.min(distance))
        return distance_norm

    # 计算误识率与拒识率函数
    def calculate_far_frr(self,distances, labels,threshold):
        same_true_count = 0  # 比较同一个人的图像时,识别正确
        same_false_count = 0  # 比较同一个人的图像时,识别错误
        different_true_count = 0  # 比较不同人的图像时,识别正确
        different_false_count = 0  # 比较不同人的图像时,识别错误

        # predict = np.empty((len(labels),))
        for i in range(len(labels)):
            if distances[i] <= threshold:
                # predict[i] = 1
                if labels[i] == 1:
                    same_true_count += 1
                else:
                    different_false_count += 1
            else:
                # predict[i] = 0
                if labels[i] == 1:
                    same_false_count += 1
                else:
                    different_true_count += 1

        # 误识率与拒识率初始化
        false_accept_rate = 0
        false_reject_rate = 0

        if different_true_count + different_false_count > 0:
            # 误识率
            false_accept_rate = different_false_count / (different_true_count + different_false_count)

        if same_true_count + same_false_count > 0:
            # 拒识率
            false_reject_rate = same_false_count / (same_true_count + same_false_count)

        return false_accept_rate,false_reject_rate





if __name__ == "__main__":
    face_rec = Face_Rec()
    #计算距离
    distance_norm = face_rec.calculate_distance(face_rec.leftfeatures,
                                    face_rec.rightfeatures, face_rec.labels)
    #     #
    joblib.dump((face_rec.leftfeatures, face_rec.rightfeatures, face_rec.labels, distance_norm),"/faceData.pkl", compress=3)
    #leftfeatures, rightfeatures, labels, distance_norm = joblib.load("/faceData.pkl")

    highestAccuracy, thres = face_rec.calculate_accuracy(distance_norm,face_rec.labels,10)
    print("highestAccuracy: ",highestAccuracy)

    false_accept_rate, false_reject_rate = face_rec.calculate_far_frr(distance_norm,face_rec.labels,0.98)
    print("false_accept_rate: ",false_accept_rate)
    print("false_reject_rate: ", false_reject_rate)
    fpr, tpr, thresholds = roc_curve(face_rec.labels, distance_norm)

    face_rec.draw_roc_curve(fpr, tpr)

