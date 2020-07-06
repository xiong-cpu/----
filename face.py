# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from skimage import io, transform
import glob
import os
#import tensorflow as tf
import numpy as np
#import time
import cv2

from numpy import linalg as LA
path = 'D:\桌面\机器学习大作业\shujuji'

# 将所有的图片resize成100*100
w = 600
h = 800
c = 1
output_dir = './faces'
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)
#   找出图片的脸
size=200
face_cascade = cv2.CascadeClassifier('E:\Anaconda3\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    #导入人脸级联分类器引擎，‘.xml’文件里包含训练出来的人脸特征
eye_cascade = cv2.CascadeClassifier('E:\Anaconda3\lib\site-packages\cv2\data\haarcascade_eye.xml')
    #导入人眼级联分类器引擎，‘.xml’文件里包含训练出来的人眼特征

def find_trace(input_img):
    global img_face
    img = io.imread(input_img)
    # img = cv2.imread(im)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    # 对，每一张脸，进行如下操作
    for (x, y, ww, hh) in faces:
        face_area = img[y:y + hh, x:x + ww]

        # 调整图片的尺寸
        img_face = cv2.resize(face_area, (size, size))
    #return img_face


# pca的本质是对图片进行降维
def pca(imgpca, k):
    w, h = imgpca.shape
    # 求均值
    for i in range(h):
        mean = np.array([np.mean(imgpca[:i])])
    # 对每一行进行零均值化
    imgpca_mean = imgpca - mean
    # 求出协方差矩阵
    cov = np.dot(np.transpose(imgpca_mean), imgpca_mean)/h
    # 计算矩阵的特征值和特征向量
    evals, evecs = LA.eig(cov)
    # 将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P
    sorted_indics = np.argsort(evals)

    topk_evecs = evecs[:, sorted_indics[:-k - 1:-1]]

    # 降维到k维后的新数据
    new = np.dot(imgpca_mean, topk_evecs)
    # 将降维后的数据映射回原空间
    img_pca_new = np.dot(new, np.transpose(topk_evecs)) + mean
    return img_pca_new


def read_img(s):
    #标签个数
    label_length = len(os.listdir(s))
    print("标签种类个数为" ,label_length)
    #数据集里的图片总数
    img_length = 0
    for i in os.listdir(path):

        full = os.path.join(path, i)
        # print(full)
        if os.path.isdir(full):
            img_len = len(os.listdir(full))
            img_length = img_length + img_len
    print("图片总数为",img_length)


    cate = [s + '/' + x for x in os.listdir(s) if os.path.isdir(s + '/' + x)]
    #print("cate is",cate)
    #imgs = [[]]
    labels = []
    #img_faces = [[]]
    img_pcas = [[]]*img_length
    #img_pcas={}
    i = 0
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.png'):
            #print('reading the images:%s' % (im))
            #img = io.imread(im)
            # img = cv2.imread(im)

            find_trace(im)
            # 提取出的人脸数据
            # img_face=face_detection(img)
            # img降维后的数据
            img_pca = pca(img_face, 12)  #对图片进行pca特征提取
            # print(img.shape)
            # print(img)
            # img=np.arange(w*h)
            # imgs.append(img)
            labels.append(idx)
            #img_faces[i] = img_face
            #print(img_pca)
            #img_pcas.append(1)
            img_pcas[i] = np.array(img_pca)
            #print(img_pcas)
            i=i+1 #将新的人脸图片放到下一个矩阵中
    # return np.asarray(imgs, np.float32), np.asarray(labels, np.int32),img_faces, img_pcas
    return img_pcas, np.asarray(labels, np.int32),label_length
    # return img_faces,np.assary(labels,np.int32)

def train_classify(x,y):
    #打乱顺序
    num_example=len(y)
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    x=x[arr]
    y=y[arr]
    #z=z[arr]
    # 将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = x[:s]
    y_train = y[:s]
    #z_train = z[:s]
    x_val = x[s:]
    y_val = y[s:]
    #z_val = z[s:]
    return x_train,y_train,x_val,y_val
#train_x, test_x, train_y, test_y =train_classify(img_pcas,label)

#print(train_x)
#print(train_y)
#print(test_x)
#print(test_y)
#print(train_x.shape)
#print(train_y.shape)

# k近邻法实现对图片比较
def k(train_pcas, labels, k, test,labellength):
    #
    l = len(labels)
    distances = [[]] * l

    # 使被识别的脸与每一张特征脸做欧式距离
    for i in range(l):
        ou = pow((test - train_pcas[i]), 2)
        distance = ou.sum(axis=1)
        distance = sum(distance)
        #print("distance is",distance)
        distances[i] = distance
    # 将待识别图片与所有pca特征脸的欧式距离从大到小排列
    distances = np.array(distances)
    sort = distances.argsort()
    count = np.arange(labellength)
    # 选出欧式距离最小的k个图片标签投票得出结果
    for i in range(k):
        vote = labels[sort[i]]
        #print("vote is",vote)
        #count.append(1)
        count[vote] = count[vote] + 1
        #print(count)
    yuce = np.argmax(count)

    return yuce

if __name__ =='__main__':
    # 获取pca处理后的图片以及标签
    pca_img,labels,labels_length=read_img(path)
    #print("pca_img的类型",type(pca_img))
    #print("labels_length is",labels_length)
    #train_pca, train_label, test_pca,test_label =train_classify(pca_img,labels)
    train_pca,test_pca, train_label, test_label = train_test_split(pca_img, labels,test_size=0.25,random_state=0)
    #对每一个测试集图片进行k近邻法预测得到预测值

    ll=len(test_label)
    yuce=np.arange(ll)
    for i in range(ll):
        #yuce.append(1)
        yuce[i]=k(train_pca,train_label,7,test_pca[i],labels_length)
    test_predict=list(yuce)
    test_labels=list(test_label)
    score = accuracy_score(test_labels,test_predict)
    print ("The accruacy socre is ", score)




