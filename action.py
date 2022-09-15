
import numpy as np
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist


predictor_path = 'E:\Anaconda\dlib_detection\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# 人脸检测器
predictor = dlib.shape_predictor(predictor_path)# 人脸特征点检测器
EYE_AR_THRESH = 0.2 # EAR眨眼的阈值
EYE_AR_CONSEC_FRAMES = 1    # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，摄像头30帧，这玩意够了
MOUTH_THRESH = 0.5# 张嘴的阈值
MOUTH_CONSEC_FRAMES = 1    # 当EAR小于阈值时，接连多少帧一定发生眨眼动作，摄像头30帧，这玩意够了
def eye_aspect_ratio(eye):
    '''
    测试眼睛的眨眼的点的纵横比
    :param eye:眼睛的坐标
    :return:眼睛的纵横比
    '''
    A = dist.euclidean(eye[1],eye[5])#计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2],eye[4])
    #水平
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    '''
    测试嘴巴的纵横比
    :param mouth: 传入嘴巴的坐标
    :return: 返回嘴巴的纵横比
    '''
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


def get_turn_point(points):
    face_point = []
    #右眼中心
    one = (points[36]+points[39])/2
    #左眼中心
    tow = (points[42]+points[45])/2
    #嘴巴中心
    three = (points[48] + points[54])/2
    face_point.append(one.astype(int))
    face_point.append(tow.astype(int))
    face_point.append(three.astype(int))
    return face_point


def get_turn_angle(face_point):
    '''
    得到脸的转向角度
    :param point_1: 左眼或右眼中心坐标
    :param point_2: 嘴巴的中心坐标
    :param point_3: 右眼或左眼中心坐标
    :return: 角度
    '''
    zuo = face_point[2][0] - face_point[0][0]
    you = face_point[1][0] - face_point[2][0]
    bei = zuo / you
    return bei



def some_action():
    # 眨眼连续帧计数
    frame_counter = 0
    # 张嘴连续帧计数
    frame_mouth_counter = 0
    # 眨眼计数
    blink_counter = 0
    # 张嘴计数
    mouth_counter = 0
    #打开内置摄像头
    cap = cv2.VideoCapture(0)
    while(1):
        #读取摄像头一帧
        ret, img = cap.read()# 读取视频流的一帧
        # 转成灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)# 人脸检测
        for rect in rects:# 遍历每一个人脸
            print('-'*20)
            shape = predictor(gray, rect)# 检测特征点
            points = face_utils.shape_to_np(shape)# convert the facial landmark (x, y)-coordinates to a NumPy array
            leftEye = points[42:47 + 1]# 取出左眼对应的特征点
            rightEye = points[36:41 + 1]# 取出右眼对应的特征点
            mouth_points = points[48:67 + 1]# 取出嘴巴对应的特征点
            #检测眨眼
            leftEAR = eye_aspect_ratio(leftEye)# 计算左眼EAR
            rightEAR = eye_aspect_ratio(rightEye)# 计算右眼EAR
            print('leftEAR = {0}'.format(leftEAR))
            print('rightEAR = {0}'.format(rightEAR))
            ear = (leftEAR + rightEAR) / 2.0# 求左右眼EAR的均值
            leftEyeHull = cv2.convexHull(leftEye)# 寻找左眼轮廓
            rightEyeHull = cv2.convexHull(rightEye)# 寻找右眼轮廓
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)# 绘制左眼轮廓
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓
            # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
            if ear < EYE_AR_THRESH:
                frame_counter += 1
            else:
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    blink_counter += 1
                frame_counter = 0
            # 在图像上显示出眨眼次数blink_counter和EAR
            cv2.putText(img, "Blinks:{0}".format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(img, "EAR:{:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


            #检测张嘴
            mouth_ear = mouth_aspect_ratio(mouth_points)
            print('MouthEAR = {0}'.format(mouth_ear))
            mouthHull = cv2.convexHull(mouth_points)  # 寻找右眼轮廓
            cv2.drawContours(img, [mouthHull], -1, (0, 255, 0), 1)  # 绘制左眼轮廓
            # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
            if mouth_ear < MOUTH_THRESH:
                frame_mouth_counter += 1
            else:
                if frame_mouth_counter >= MOUTH_CONSEC_FRAMES:
                    mouth_counter += 1
                frame_mouth_counter = 0

            # 在图像上显示出眨眼次数blink_counter和EAR
            cv2.putText(img, "Open:{0}".format(mouth_counter), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                        2)
            cv2.putText(img, "MOUTH_EAR:{:.2f}".format(mouth_ear), (300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            #脸部右转
            face_point = get_turn_point(points)
            bei = get_turn_angle(face_point) #大于1是右向，小于1是左向，设置右向阈值为3，左向阈值为0.5

            cv2.putText(img, "bei:{0}".format(bei), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                        2)



        #展示摄像头画面
        cv2.imshow("Frame", img)
        #按q退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     some_action()

