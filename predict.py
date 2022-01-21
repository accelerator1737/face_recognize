import numpy as np
from PIL import Image
from imageio import imread
from facenet import Facenet
import dlib
import math
import cv2

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
    # 使用特征提取器get_frontal_face_detector
    detector = dlib.get_frontal_face_detector()

    # dlib的68点模型，使用作者训练好的特征预测器
    predictor_path = 'E:\Anaconda\dlib_detection\shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

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


if __name__ == "__main__":
    model = Facenet()
    while True:
        path = input('Input image filename:')
        try:
            # path = r'E:\人脸识别\数据集\data\0000045\001.jpg'
            image_1 = Image.open(path)
            i1 = imread(path)
        except:
            print('Image Open Error! Try again!')
            continue
        model.detect_image(i1)





