import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from imageio import imread
import dlib
import cv2
import os
from nets.facenet import Facenet as facenet
from to_database import MSSQL, get_path, save_picture


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
#--------------------------------------------#

# 使用特征提取器get_frontal_face_detector
detector = dlib.get_frontal_face_detector()

# dlib的68点模型，使用作者训练好的特征预测器
predictor_path = 'E:\Anaconda\dlib_detection\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

img_size = 160
#得到旋转的角度
def get_angel(scope):
    '''
    得到图像旋转的角度
    :param scope: 人脸68个关键点
    :return: 旋转的角度
    '''
    #左边眼睛中心的坐标
    left_x = (scope.part(36).x + scope.part(39).x) / 2
    left_y = (scope.part(36).y + scope.part(39).y) / 2
    # 右边眼睛中心的坐标
    right_x = (scope.part(42).x + scope.part(45).x) / 2
    right_y = (scope.part(42).y + scope.part(45).y) / 2
    sin = math.sin((right_y-left_y) / (math.sqrt((left_x-right_x)**2 + (left_y-right_y)**2)))
    cos = math.cos((right_x-left_x) / (math.sqrt((left_x-right_x)**2 + (left_y-right_y)**2)))
    angel = math.atan(sin / cos) * 180 / math.pi
    return angel


def geometry_normalize(scope, img_matrix):
    '''
    进行几何归一化
    :param scope: 脸的68个点位置
    :param img_matrix: 传入原图片的矩阵
    :return: 返回一个归一化的图片的矩阵
    '''
    rows, cols, channels = img_matrix.shape
    #左边眼睛中心的坐标
    left_x = (scope.part(36).x + scope.part(39).x) / 2
    left_y = (scope.part(36).y + scope.part(39).y) / 2
    # 右边眼睛中心的坐标
    right_x = (scope.part(42).x + scope.part(45).x) / 2
    right_y = (scope.part(42).y + scope.part(45).y) / 2
    # 两眼中心的坐标
    mid_x = (left_x + right_x) / 2
    mid_y = (left_y + right_y) / 2
    #两眼的距离
    d = right_x - left_x
    left_top = [mid_x - d if mid_x - d > 0 else 0, mid_y - 0.5 * d if mid_y - 0.5 * d > 0 else 0]
    right_bottom = [mid_x + d if mid_x - d < cols else cols, mid_y + 1.5 * d if mid_y + 1.5 * d < rows else rows]
    if scope.part(9).y > right_bottom[1]:
        right_bottom[1] = scope.part(9).y
    if scope.part(0).x < left_top[0]:
        left_top[0] = scope.part(0).x
    if scope.part(16).x > right_bottom[0]:
        right_bottom[0] = scope.part(16).x
    return left_top, right_bottom


def get_resize(img):
    '''
    将图片转为64*64矩阵
    :param img: 读入的图片矩阵
    :return: 是否是合适的人脸、几何归一化后的64*64 3通道矩阵
    '''
    global detector, predictor
    # 特征提取器的实例化
    dets = detector(img)
    flag = 0
    gehe = 0

    if not dets:
        return flag, gehe
    # 对图片中的一张人脸进行几何归一化
    d = dets[0]
    shape = predictor(img, d)

    left, right = geometry_normalize(shape, img)

    gehe = img[int(left[1]):int(right[1]), int(left[0]):int(right[0]), :]
    gehe = cv2.resize(gehe, (img_size, img_size))
    flag = 1
    #得到灰度图
    return flag, gehe

class Facenet(object):
    _defaults = {
        "model_path"    : "model_data/facenet_mobilenet.pth",
        "input_shape"   : (160, 160, 3),
        "backbone"      : "mobilenet",
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化Facenet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()
        
    def generate(self):
        # 载入模型
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = facenet(backbone=self.backbone, mode="predict")
        model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = model.eval()

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
            
        print('{} model loaded.'.format(self.model_path))
    

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_1):
        #---------------------------------------------------#
        #   图片预处理，归一化
        #---------------------------------------------------#
        #实例化数据库
        ms = MSSQL(host="127.0.0.1:1433", user="sa", pwd="123456", db="SuperMarket")
        L = get_path(ms)
        all_value = []
        all_name = []
        with torch.no_grad():
            flag, image_1 = get_resize(image_1)
            if not flag:
                print('没有找到人脸')
                return False, 1 ,1
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.asarray(image_1).astype(np.float64)/255,(2,0,1)),0)).type(torch.FloatTensor)

            if self.cuda:
                photo_1 = photo_1.cuda()
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
        img = []
        for i in L:
            a = i.split('\\')
            name = a[-2]
            image_2 = imread(i)
            flag, image_2 = get_resize(image_2)
            if not flag:
                continue
            img.append(image_2)
            with torch.no_grad():
                photo_2 = torch.from_numpy(
                    np.expand_dims(np.transpose(np.asarray(image_2).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
                    torch.FloatTensor)

                if self.cuda:
                    photo_2 = photo_2.cuda()
                    # ---------------------------------------------------#
                #   图片传入网络进行预测
                # ---------------------------------------------------#
                output2 = self.net(photo_2).cpu().numpy()
            #---------------------------------------------------#
            #   计算二者之间的距离
            #---------------------------------------------------#
            l1 = np.linalg.norm(output1-output2, axis=1)
            all_value.append(l1[0])
            all_name.append(name)
        index1 = all_value.index(min(all_value))
        print(min(all_value))
        people = all_name[index1]
        print(people)
        return True, people, min(all_value)
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))
        #
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(img[index1]))
        # plt.text(-12, -12, 'Distance:%.3f' % min(all_value), ha='center', va='bottom', fontsize=11)
        # plt.show()




