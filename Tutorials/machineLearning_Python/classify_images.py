#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/3/16 12:04
@Author  : cai

教程来自 https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os


def extract_color_stats(image):
    '''
    将图片分成 RGB 三通道，然后分别计算每个通道的均值和标准差，然后返回
    :param image:
    :return:
    '''
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]

    return features


# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="3scenes",
                help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="knn",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())

# 定义一个保存模型的字典，根据 key 来选择加载哪个模型
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="rbf", gamma="auto"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}
# 加载数据并提取特征
print("[INFO] extracting image features...")
imagePaths = paths.list_images(args['dataset'])
data = []
labels = []

# 循环遍历所有的图片数据
for imagePath in imagePaths:
    # 加载图片，然后计算图片的颜色通道统计信息
    image = Image.open(imagePath)
    features = extract_color_stats(image)
    data.append(features)

    # 保存图片的标签信息
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# 对标签进行编码，从字符串变为整数类型
le = LabelEncoder()
labels = le.fit_transform(labels)

# 进行训练集和测试集的划分，75%数据作为训练集，其余25%作为测试集
(trainX, testX, trainY, testY) = train_test_split(data, labels, random_state=3, test_size=0.25)
# print('trainX numbers={}, testX numbers={}'.format(len(trainX), len(testX)))

# 训练模型
print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

# 预测并输出分类结果报告
print("[INFO] evaluating...")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=le.classes_))

