#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/3/16 12:05
@Author  : cai

教程来自 https://www.pyimagesearch.com/2019/01/14/machine-learning-in-python/
"""
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris

# 载入 Iris 数据集，然后进行训练集和测试集的划分，75%数据作为训练集，其余25%作为测试集
print("[INFO] loading data...")
dataset = load_iris()
(trainX, testX, trainY, testY) = train_test_split(dataset.data,
                                                  dataset.target, test_size=0.25)

# 将标签进行 one-hot 编码
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 利用 Keras 定义网络模型
model = Sequential()
model.add(Dense(3, input_shape=(4,), activation="sigmoid"))
model.add(Dense(3, activation="sigmoid"))
model.add(Dense(3, activation="softmax"))

# 采用梯度下降训练模型
print('[INFO] training network...')
opt = SGD(lr=0.1, momentum=0.9, decay=0.1 / 250)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=250, batch_size=16)

# 预测
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=16)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=dataset.target_names))
