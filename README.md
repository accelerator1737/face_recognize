# face_recognize

### 项目描述

该项目是基于人脸识别而做成的上班打卡的系统，系统可以读取数据库中的图片与信息，在进行打卡识别的时候，系统会先让摄像头前的人进行眨左眼、眨右眼、张嘴、脸部左转、右转，确保摄像头前的人是本人，在进行人脸识别。人脸识别时，系统会对摄像头前的人进行拍摄，依靠68个点的模型，提取出人的68个特征点，然后计算出照片中人的各点的欧氏距离，将这个欧氏距离与数据库中的每个人的照片的欧式距离相比对，欧氏距离小的两张图片即匹配，匹配过程中还应该设置一个阈值，若欧氏距离大于这个阈值，则认为数据库中没有这个人。

### 项目实现

该项目连接的数据库是SQL Serve，使用的模型有68个特征点的提取模型、人脸检测模型以及训练的模型。在模型训练中所运用的是pytorch搭建的卷积神经网络。

### 环境需求

