#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/3/14 23:12
@Author  : cai

模型的性能度量，二分类
模型评估方法，划分数据集
"""
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


# =======================================
# 性能度量方法
# =======================================

def accuracy(y_true, y_pred):
    return sum(y == y_p for y, y_p in zip(y_true, y_pred)) / len(y_true)


def error(y_true, y_pred):
    return sum(y != y_p for y, y_p in zip(y_true, y_pred)) / len(y_true)


def precision(y_true, y_pred):
    true_positive = sum(y and y_p for y, y_p in zip(y_true, y_pred))
    predicted_positive = sum(y_pred)
    return true_positive / predicted_positive


def recall(y_true, y_pred):
    true_positive = sum(y and y_p for y, y_p in zip(y_true, y_pred))
    real_positive = sum(y_true)
    return true_positive / real_positive


def true_negative_rate(y_true, y_pred):
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y_true, y_pred))
    actual_negative = len(y_true) - sum(y_true)
    return true_negative / actual_negative


def roc(y, y_hat_prob):
    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([recall(y, y_hat), 1 - true_negative_rate(y, y_hat)])
    return ret


def get_auc(y, y_hat_prob):
    roc_val = iter(roc(y, y_hat_prob))
    tpr_pre, fpr_pre = next(roc_val)
    auc = 0
    for tpr, fpr in roc_val:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        tpr_pre = tpr
        fpr_pre = fpr
    return auc


# =======================================
# 模型评估方法
# =======================================

def hold_out():
    # 加载 Iris 数据集
    dataset = load_iris()
    # 划分训练集和测试集
    (trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, random_state=3, test_size=0.3)
    # 建立模型
    knn = KNeighborsClassifier()
    # 训练模型
    knn.fit(trainX, trainY)
    # 将准确率打印出
    print('hold_out, score:', knn.score(testX, testY))


def cross_validation(cv=10):
    # 加载 Iris 数据集
    dataset = load_iris()
    data = dataset.data
    label = dataset.target
    # 建立模型
    knn = KNeighborsClassifier()
    # 使用K折交叉验证模块
    scores = cross_val_score(knn, data, label, cv=cv, scoring='accuracy')
    print('cross_validation numbers=', cv)
    # 将每次的预测准确率打印出
    print(scores)
    # 将预测准确平均率打印出
    print(scores.mean())


def StratifiedKFold_method(n_splits=3):
    '''
    分层采样
    :return:
    '''
    # 加载 Iris 数据集
    dataset = load_iris()
    data = dataset.data
    label = dataset.target
    # 建立模型
    knn = KNeighborsClassifier()
    print('use StratifiedKFold')
    skfolds = StratifiedKFold(n_splits=n_splits, random_state=42)
    scores = 0.
    for train_index, test_index in skfolds.split(data, label):
        clone_clf = clone(knn)
        X_train_folds = data[train_index]
        y_train_folds = (label[train_index])
        X_test_fold = data[test_index]
        y_test_fold = (label[test_index])
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
        scores += n_correct / len(y_pred)
    print('mean scores:', scores / n_splits)


if __name__ == '__main__':
    y_true = [1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0]
    y_hat_prob = [0.9, 0.85, 0.8, 0.7, 0.6]

    acc = accuracy(y_true, y_pred)
    err = error(y_true, y_pred)
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    print('accuracy=', acc)
    print('error=', err)
    print('precisions=', precisions)
    print('recalls=', recalls)

    roc_list = roc(y_true, y_hat_prob)
    auc_val = get_auc(y_true, y_hat_prob)
    print('roc_list:', roc_list)
    print('auc_val:', auc_val)

    hold_out()
    cross_validation()
    StratifiedKFold_method()
