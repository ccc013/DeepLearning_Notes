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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
import argparse

# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn", help="type of python machine learning model to use")
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

# 载入 Iris 数据集，然后进行训练集和测试集的划分，75%数据作为训练集，其余25%作为测试集
print("[INFO] loading data...")
dataset = load_iris()
(trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, random_state=3, test_size=0.25)

# 训练模型
print("[INFO] using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

# 预测并输出一份分类结果报告
print("[INFO] evaluating")
predictions = model.predict(testX)
print(classification_report(testY, predictions, target_names=dataset.target_names))

