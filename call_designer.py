import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from untitled import *
import cv2
from information import *
import time
from action import *
import copy
from facenet import Facenet
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
from connection import *
from enter import *
from imageio import imread, imsave
import os
from generate_table import *
from emergency import *
now_img = ''
cap1 = cv2.VideoCapture(0)


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('人脸识别打卡')
        #定义定时器
        # self.timer_camera = QTimer(self)
        # #cv2调用内置摄像头
        # self.cap = cv2.VideoCapture(0)
        # #每10ms调用一次获取帧
        # self.timer_camera.timeout.connect(self.sign_in)
        # self.timer_camera.start(20)
        self.pushButton.clicked.connect(self.sign_in)

        self.model = Facenet()
        self.take_card = 0
        # 眨眼连续帧计数
        self.frame_counter = 0
        # 张嘴连续帧计数
        self.frame_mouth_counter = 0
        # 歪头连续帧计数
        self.frame_head_counter = 0
        # 眨眼计数
        self.blink_counter = 0
        # 张嘴计数
        self.mouth_counter = 0
        # 歪头计数
        self.head_counter = 0
        self.pushButton_2.clicked.connect(self.goto_1)
        self.dic = {}
        self.ms = MSSQL(host="127.0.0.1:1433",user="sa",pwd="123456",db="SuperMarket")
        r1 = get_information(self.ms)
        for i in r1:
            self.dic[i[0]] = int(i[2])


        show = imread('F:\图片\CC10586F338984590515A994336018EF.jpg')
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))  # 将摄像头显示在之前创建的Label控件中
        self.label.setScaledContents(True)

    def goto_1(self):
        # global cap1
        # cap1.release()
        # self.cap.release()
        # print(self.cap.isOpened())
        # self.cap = 0
        # del self.cap
        self.close()



    def add_chinese(self, img_OpenCV, str, position, fillColor):
        '''
        添加中文入图片
        :param show: OpenCV读取的照片格式
        :param person_name: 要添加的中文
        :param position: 要添加的中文的位置
        :param color: 要添加的中文的颜色
        :return: OpenCV格式
        '''
        location = r'C:\Windows\Fonts\simkai.ttf'
        # 图像从OpenCV格式转换成PIL格式
        img_PIL = Image.fromarray(img_OpenCV)

        # 字体 字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
        font = ImageFont.truetype(location, 40)


        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, str, font=font, fill=fillColor)
        # 转换回OpenCV格式
        # img_OpenCV = cv2.cvtColor(np.asarray(img_PIL))
        return np.asarray(img_PIL)


    def sign_in(self):
        '''
        显示cv2电泳摄像头获取的帧
        :return:
        '''
        # global cap1
        self.cap = cv2.VideoCapture(0)
        self.take_card = random.randint(1, 2)
        while True:
            success, frame = self.cap.read()
            # success, frame = cap1.read()
            if success:
                show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                temp = copy.deepcopy(show)
                # 转成灰度图像
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)  # 人脸检测
                #遍历每个人脸
                for rect in rects:  # 遍历每一个人脸
                    shape = predictor(gray, rect)  # 检测特征点
                    # convert the facial landmark (x, y)-coordinates to a NumPy array
                    points = face_utils.shape_to_np(shape)
                    cv2.rectangle(show, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
                    #开始识别

                    flag, person_name, shrold = self.model.detect_image(temp)
                    if flag:
                        if shrold <= 0.9:
                            if person_name:
                                #添加中文
                                show = self.add_chinese(show, person_name, (rect.left(), rect.top()), (0, 255, 0))
                            #眨眼
                            if self.take_card == 1:
                                show = self.add_chinese(show, '请眨眼', (10, 30), (0, 255, 0))
                                leftEye = points[42:47 + 1]  # 取出左眼对应的特征点
                                rightEye = points[36:41 + 1]  # 取出右眼对应的特征点
                                # 检测眨眼
                                leftEAR = eye_aspect_ratio(leftEye)  # 计算左眼EAR
                                rightEAR = eye_aspect_ratio(rightEye)  # 计算右眼EAR
                                ear = (leftEAR + rightEAR) / 2.0  # 求左右眼EAR的均值
                                # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
                                if ear < EYE_AR_THRESH:
                                    self.frame_counter += 1
                                else:
                                    if self.frame_counter >= EYE_AR_CONSEC_FRAMES:
                                        self.blink_counter += 1
                                    self.frame_counter = 0
                                if self.blink_counter >= 1:
                                    print('打卡成功')
                                    showMessage = QMessageBox.information
                                    showMessage(self, '提示', "{}打卡成功，时间{}".format(person_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())), QMessageBox.Yes)
                                    ztime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    date = ztime.split()[0]
                                    time_now = ztime.split()[1]
                                    self.ms.ExecNonQuery("insert into sign_in(name,ID,go_ahead,come) values('{}', {},'{}','{}')".format(person_name,
                                                    self.dic[person_name], date, time_now))
                                    self.blink_counter = 0
                                    self.take_card = 0
                            #检测张嘴
                            elif self.take_card == 2:
                                show = self.add_chinese(show, '请张嘴', (10, 30), (0, 255, 0))
                                mouth_points = points[48:67 + 1]  # 取出嘴巴对应的特征点
                                mouth_ear = mouth_aspect_ratio(mouth_points)
                                # 如果EAR小于阈值，开始计算连续帧，只有连续帧计数超过EYE_AR_CONSEC_FRAMES时，才会计做一次眨眼
                                if mouth_ear < MOUTH_THRESH:
                                    self.frame_mouth_counter += 1
                                else:
                                    if self.frame_mouth_counter >= MOUTH_CONSEC_FRAMES:
                                        self.mouth_counter += 1
                                    self.frame_mouth_counter = 0
                                if self.mouth_counter >= 1:
                                    print('打卡成功')

                                    showMessage = QMessageBox.information
                                    showMessage(self, '提示',
                                                "{}打卡成功，时间{}".format(person_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                                                QMessageBox.Yes)
                                    ztime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    date = ztime.split()[0]
                                    time_now = ztime.split()[1]
                                    self.ms.ExecNonQuery(
                                        "insert into sign_in(name,ID,go_ahead,come) values('{}', {},'{}','{}')".format(person_name,
                                                                                                                        self.dic[
                                                                                                                            person_name],
                                                                                                                        date,
                                                                                                                        time_now))
                                    self.mouth_counter = 0
                                    self.take_card = 0
                            #右歪头
                            elif self.take_card == 3:
                                show = self.add_chinese(show, '请向右歪头', (10, 30), (0, 255, 0))
                                face_point = get_turn_point(points)
                                bei = get_turn_angle(face_point)  # 大于1是右向，小于1是左向，设置右向阈值为3，左向阈值为0.5
                                if bei >= 3:
                                    self.frame_head_counter += 1
                                else:
                                    if self.frame_head_counter >= 2:
                                        self.head_counter += 1
                                    self.frame_head_counter = 0
                                if self.head_counter >= 1:
                                    print('打卡成功')
                                    showMessage = QMessageBox.information
                                    showMessage(self, '提示',
                                                "{}打卡成功，时间{}".format(person_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                                                QMessageBox.Yes)
                                    ztime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    date = ztime.split()[0]
                                    time_now = ztime.split()[1]
                                    self.ms.ExecNonQuery(
                                        "insert into sign_in(name,ID,go_ahead,come) values('{}', {},'{}','{}')".format(person_name,
                                                                                                                        self.dic[
                                                                                                                            person_name],
                                                                                                                        date,
                                                                                                                        time_now))
                                    self.head_counter = 0
                                    self.take_card = 0
                            # 左歪头
                            elif self.take_card == 4:
                                show = self.add_chinese(show, '请向左歪头', (10, 30), (0, 255, 0))
                                face_point = get_turn_point(points)
                                bei = get_turn_angle(face_point)  # 大于1是右向，小于1是左向，设置右向阈值为3，左向阈值为0.5
                                if bei <= 0.5:
                                    self.frame_head_counter += 1
                                else:
                                    if self.frame_head_counter >= 2:
                                        self.head_counter += 1
                                    self.frame_head_counter = 0
                                if self.head_counter >= 1:
                                    print('打卡成功')
                                    showMessage = QMessageBox.information
                                    showMessage(self, '提示',
                                                "{}打卡成功，时间{}".format(person_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                                                QMessageBox.Yes)
                                    ztime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    date = ztime.split()[0]
                                    time_now = ztime.split()[1]
                                    self.ms.ExecNonQuery(
                                        "insert into sign_in(name,ID,go_ahead,come) values('{}', {},'{}','{}')".format(person_name,
                                                                                                                        self.dic[
                                                                                                                            person_name],
                                                                                                                        date,
                                                                                                                        time_now))
                                    self.head_counter = 0
                                    self.take_card = 0


                showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.label.setPixmap(QPixmap.fromImage(showImage))  # 将摄像头显示在之前创建的Label控件中
                cv2.waitKey(1000)
                #重新计时
                # self.timer_camera.start(20)
                #图片适应QLabel的大小
                # self.label.setScaledContents(True)


    def exchange_card(self):    #1是眨眼，2是张嘴，3是右歪头，4是左歪头
        self.take_card = random.randint(1, 2)


    # def goto_1(self):
    #     # self.timer_camera.timeout.disconnect(self.test)
    #     child1 = ChildWindow1()
    #     child1.show()


class ChildWindow1(QDialog, Ui_Dialog):
    def __init__(self):
        super(ChildWindow1, self).__init__()
        self.setupUi(self)
        self.ms = MSSQL(host="127.0.0.1:1433",user="sa",pwd="123456",db="SuperMarket")
        self.setWindowTitle('数据管理')
        self.data = get_information(self.ms)
        self.tableWidget.setRowCount(len(self.data))
        self.tableWidget.setColumnCount(len(self.data[0]))
        self.tableWidget.setHorizontalHeaderLabels(["姓名", "图片", "人员ID", '性别', '手机号码'])
        for i, (name, tu, ren, gender, tele) in enumerate(self.data):
            item_name = QTableWidgetItem(name)
            item_tu = QTableWidgetItem(tu)
            item_ren = QTableWidgetItem(ren)
            item_gender = QTableWidgetItem(gender)
            item_tele = QTableWidgetItem(tele)
            self.tableWidget.setItem(i, 0, item_name)
            self.tableWidget.setItem(i, 1, item_tu)
            self.tableWidget.setItem(i, 2, item_ren)
            self.tableWidget.setItem(i, 3, item_gender)
            self.tableWidget.setItem(i, 4, item_tele)
        self.pushButton.clicked.connect(self.add_infor)  # 录入信息
        self.pushButton_2.clicked.connect(self.delete_table)    #删除信息
        # self.pushButton_3.clicked.connect(self.generate_table)  # 生成报表
        self.pushButton_5.clicked.connect(self.save_infor)  # 保存信息
        self.pushButton_6.clicked.connect(self.upload)  # 更新信息
        # self.pushButton.clicked.connect(self.goto_1)
        # self.pushButton_3.clicked.connect(self.goto_2)


    # def goto_2(self):
    #     child3 = ChildWindow3()
    #     child3.show()
    #
    #
    # def goto_1(self):
    #     child2 = ChildWindow2()
    #     child2.show()


    def add_infor(self):
        child2 = ChildWindow2()
        child2.show()
        child2.exec_()


    def upload(self):
        '''
        更新信息
        :return:
        '''
        self.data = get_information(self.ms)
        self.tableWidget.setRowCount(len(self.data))
        self.tableWidget.setColumnCount(len(self.data[0]))
        for i, (name, tu, ren, gender, tele) in enumerate(self.data):
            item_name = QTableWidgetItem(name)
            item_tu = QTableWidgetItem(tu)
            item_ren = QTableWidgetItem(ren)
            item_gender = QTableWidgetItem(gender)
            item_tele = QTableWidgetItem(tele)
            self.tableWidget.setItem(i, 0, item_name)
            self.tableWidget.setItem(i, 1, item_tu)
            self.tableWidget.setItem(i, 2, item_ren)
            self.tableWidget.setItem(i, 3, item_gender)
            self.tableWidget.setItem(i, 4, item_tele)


    def save_infor(self):
        '''
        保存信息
        :return:
        '''
        self.ms.ExecNonQuery('TRUNCATE TABLE face')
        # 获取表格行数和列数
        row_num = self.tableWidget.rowCount()
        cols_num = self.tableWidget.columnCount()
        table = []
        # 存储表格数值
        for i in range(0, row_num):
            tabel_zi = []
            for j in range(0, cols_num):
                tabel_zi.append(self.tableWidget.item(i, j).text())
            table.append(tabel_zi)
        for i in range(len(table)):
            self.ms.ExecNonQuery("insert into face(name,path,ID,gender,telephone) values('{}', '{}',{},'{}','{}')".format(table[i][0],
                                        table[i][1], int(table[i][2]), table[i][3], table[i][4]))



    def closeEvent(self, event):
        '''
        拦截窗口关闭事件
        :param event:
        :return:
        '''
        showMessage = QMessageBox.question
        reply = showMessage(self, '警告', "系统将退出，是否保存?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.save_infor()
            event.accept()
        else:
            event.accept()


    def delete_table(self):
        '''
        删除一行信息
        :return:
        '''
        indices = self.tableWidget.selectionModel().selectedRows()
        for index in sorted(indices):
            self.tableWidget.removeRow(index.row())


class ChildWindow2(QDialog, Ui_Dialog2):
    def __init__(self):
        super(ChildWindow2, self).__init__()
        self.setupUi(self)
        self.timer_camera = QTimer(self)
        #cv2调用内置摄像头
        print(cap1.isOpened())
        self.cap = cap1
        # self.cap1 = cv2.VideoCapture(0)
        #每10ms调用一次获取帧

        self.timer_camera.timeout.connect(self.go)
        self.timer_camera.start(20)
        self.pushButton.clicked.connect(self.get_sth)


    def get_sth(self):
        global now_img
        persons_name = self.lineEdit.text()
        persons_gender = self.lineEdit_2.text()
        persons_id = self.lineEdit_3.text()
        persons_tele = self.lineEdit_4.text()
        ms = MSSQL(host="127.0.0.1:1433",user="sa",pwd="123456",db="SuperMarket")
        r = remove_blank(ms.ExecQuery('select ID from face'))
        if persons_id in r:
            showMessage = QMessageBox.question
            showMessage(self, '警告', "该员工号已被占有，请换一个", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            path = 'E:\人脸识别\数据集\data' + '\\' + persons_name
            if not os.path.exists(path):
                os.mkdir(path)
            temp_name = persons_name + '_' + str(len(os.listdir(path)) + 100) + '.jpg'
            path = path + '\\' + temp_name
            imsave(path, now_img)
            #关闭窗口
            ms.ExecNonQuery("insert into face(name,path,ID,gender,telephone) values('{}', '{}',{},'{}','{}')".format(persons_name,
                                        path, int(persons_id), persons_gender, persons_tele))
            self.close()


    def go(self):
        global now_img
        success, frame = self.cap.read()
        if success:
            show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now_img = show
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))  # 将摄像头显示在之前创建的Label控件中
        #重新计时
        self.timer_camera.start(20)


class ChildWindow3(QDialog, Ui_Dialog3):
    def __init__(self):
        super(ChildWindow3, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.get_year)
        self.pushButton_2.clicked.connect(self.get_month)
        self.pushButton_3.clicked.connect(self.get_day)
        self.pushButton_4.clicked.connect(self.get_query)
        self.ms = MSSQL(host="127.0.0.1:1433",user="sa",pwd="123456",db="SuperMarket")
        self.pushButton_5.clicked.connect(self.get_all)


    def get_all(self):
        r = self.ms.ExecQuery('select * from sign_in')
        tu = []
        for i in r:
            tu.append(remove_infor_blank(i))
        j = 0
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(["员工姓名", "员工ID", "出勤日期", '出勤时间'])
        self.tableWidget.setRowCount(len(tu))
        for i in tu:
            item_name = QTableWidgetItem(i[0])
            item_id = QTableWidgetItem(i[1])
            item_date = QTableWidgetItem(i[2])
            item_time = QTableWidgetItem(i[3])
            self.tableWidget.setItem(j, 0, item_name)
            self.tableWidget.setItem(j, 1, item_id)
            self.tableWidget.setItem(j, 2, item_date)
            self.tableWidget.setItem(j, 3, item_time)
            j += 1


    def get_query(self):
        query_name = self.lineEdit.text()
        r2 = self.timeEdit.time().toString('hh:mm') #止
        r1 = self.timeEdit_2.time().toString('hh:mm')   #起
        # 获取表格行数和列数
        row_num = self.tableWidget.rowCount()
        cols_num = self.tableWidget.columnCount()
        table = []
        # 存储表格数值
        for i in range(0, row_num):
            tabel_zi = []
            for j in range(0, cols_num):
                tabel_zi.append(self.tableWidget.item(i, j).text())
            table.append(tabel_zi)
        tu = []
        if query_name:
            for i in table:
                if i[0] == query_name and r1 <= i[3] <= r2:
                    tu.append(i)
        else:
            for i in table:
                if r1 <= i[3] <= r2:
                    tu.append(i)
        j = 0
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(["员工姓名", "员工ID", "出勤日期", '出勤时间'])
        self.tableWidget.setRowCount(len(tu))
        for i in tu:
            item_name = QTableWidgetItem(i[0])
            item_id = QTableWidgetItem(i[1])
            item_date = QTableWidgetItem(i[2])
            item_time = QTableWidgetItem(i[3])
            self.tableWidget.setItem(j, 0, item_name)
            self.tableWidget.setItem(j, 1, item_id)
            self.tableWidget.setItem(j, 2, item_date)
            self.tableWidget.setItem(j, 3, item_time)
            j += 1



    def get_year(self):
        s = self.dateEdit.date().toString(Qt.ISODate)
        year_para = s.split('-')[0]
        r = self.ms.ExecQuery("select * from sign_in where year(go_ahead)={}".format(year_para))
        tu = []
        for i in r:
            tu.append(remove_infor_blank(i))
        j = 0
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(["员工姓名", "员工ID", "出勤日期", '出勤时间'])
        self.tableWidget.setRowCount(len(tu))
        for i in tu:
            item_name = QTableWidgetItem(i[0])
            item_id = QTableWidgetItem(i[1])
            item_date = QTableWidgetItem(i[2])
            item_time = QTableWidgetItem(i[3])
            self.tableWidget.setItem(j, 0, item_name)
            self.tableWidget.setItem(j, 1, item_id)
            self.tableWidget.setItem(j, 2, item_date)
            self.tableWidget.setItem(j, 3, item_time)
            j += 1
        print(tu)


    def get_month(self):
        s = self.dateEdit.date().toString(Qt.ISODate)
        year_para = s.split('-')[0]
        month_para = s.split('-')[1]
        r = self.ms.ExecQuery("select * from sign_in where  DATEPART(year,go_ahead)={} and DATEPART(month,go_ahead)={}".format(year_para, month_para))
        tu = []
        for i in r:
            tu.append(remove_infor_blank(i))
        j = 0
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(["员工姓名", "员工ID", "出勤日期", '出勤时间'])
        self.tableWidget.setRowCount(len(tu))
        for i in tu:
            item_name = QTableWidgetItem(i[0])
            item_id = QTableWidgetItem(i[1])
            item_date = QTableWidgetItem(i[2])
            item_time = QTableWidgetItem(i[3])
            self.tableWidget.setItem(j, 0, item_name)
            self.tableWidget.setItem(j, 1, item_id)
            self.tableWidget.setItem(j, 2, item_date)
            self.tableWidget.setItem(j, 3, item_time)
            j += 1


    def get_day(self):
        s = self.dateEdit.date().toString(Qt.ISODate)
        year_para = s.split('-')[0]
        month_para = s.split('-')[1]
        day_para = s.split('-')[2]
        r = self.ms.ExecQuery("select * from sign_in where  DATEPART(year,go_ahead)={} and DATEPART(month,go_ahead)={} and DATEPART(day,go_ahead)={}".format(year_para, month_para, day_para))
        tu = []
        for i in r:
            tu.append(remove_infor_blank(i))
        j = 0
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(["员工姓名", "员工ID", "出勤日期", '出勤时间'])
        self.tableWidget.setRowCount(len(tu))
        for i in tu:
            item_name = QTableWidgetItem(i[0])
            item_id = QTableWidgetItem(i[1])
            item_date = QTableWidgetItem(i[2])
            item_time = QTableWidgetItem(i[3])
            self.tableWidget.setItem(j, 0, item_name)
            self.tableWidget.setItem(j, 1, item_id)
            self.tableWidget.setItem(j, 2, item_date)
            self.tableWidget.setItem(j, 3, item_time)
            j += 1


class ChildWindow4(QDialog, Ui_Dialog4):
    def __init__(self):
        super(ChildWindow4, self).__init__()
        self.setupUi(self)
        # 定义定时器
        self.timer_camera = QTimer(self)
        # cv2调用内置摄像头
        # self.cap = cv2.VideoCapture(0)
        # 每10ms调用一次获取帧
        # self.timer_camera.timeout.connect(self.sign_in)
        self.timer_camera.timeout.connect(self.test)
        self.timer_camera.start(20)


    def test(self):
        # success, frame = self.cap.read()
        # if success:
        #     show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        #     self.label.setPixmap(QPixmap.fromImage(showImage))  # 将摄像头显示在之前创建的Label控件中
        # # 重新计时
        # self.timer_camera.start(20)
        pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MyWindow()

    # 实例化数据管理子窗口
    child1 = ChildWindow1()
    # 主窗体按钮事件绑定
    btn = main.pushButton_2
    btn.clicked.connect(child1.show)

    # child2 = ChildWindow2()
    # # 主窗体按钮事件绑定
    # btn1 = child1.pushButton
    # btn1.clicked.connect(child2.show)

    child3 = ChildWindow3()
    btn2 = child1.pushButton_3
    btn2.clicked.connect(child3.show)

    #打开摄像头
    # child4 = ChildWindow4()
    # btn3 = main.pushButton
    # btn3.clicked.connect(child4.show)

    main.show()
    sys.exit(app.exec_())

