#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/1/23 23:09
@Author  : cai

practise for different loss function
"""
import numpy as np


def rmse(predictions, targets):
    # 真实值和预测值的误差
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    # 取平方根
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val


def mae(predictions, targets):
    differences = predictions - targets
    absolute_differences = np.absolute(differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences


def mbe(predictions, targets):
    differences = predictions - targets
    mean_absolute_differences = differences.mean()
    return mean_absolute_differences


def hinge_loss(predictions, label):
    '''
    hinge_loss = max(0, s_j - s_yi +1)
    :param predictions:
    :param label:
    :return:
    '''
    result = 0.0
    pred_value = predictions[label]
    print('pred_value={}'.format(pred_value))
    for i, val in enumerate(predictions):
        if i == label:
            continue
        tmp = val - pred_value + 1
        result += max(0, tmp)
    return result


def cross_entropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5))) / N
    return ce_loss


def loss_test():
    y_hat = np.array([0.000, 0.166, 0.333])
    y_true = np.array([0.000, 0.254, 0.998])

    print("d is: " + str(["%.8f" % elem for elem in y_hat]))
    print("p is: " + str(["%.8f" % elem for elem in y_true]))
    rmse_val = rmse(y_hat, y_true)
    print("rms error is: " + str(rmse_val))
    mae_val = mae(y_hat, y_true)
    print("mae error is: " + str(mae_val))

    mbe_val = mbe(y_hat, y_true)
    print("mbe error is: " + str(mbe_val))

    image1 = np.array([-0.39, 1.49, 4.21])
    image2 = np.array([-4.61, 3.28, 1.46])
    image3 = np.array([1.03, -2.37, -2.27])
    result1 = hinge_loss(image1, 0)
    result2 = hinge_loss(image2, 1)
    result3 = hinge_loss(image3, 2)
    print('image1,hinge loss={}'.format(result1))
    print('image2,hinge loss={}'.format(result2))
    print('image3,hinge loss={}'.format(result3))

    predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                            [0.01, 0.01, 0.01, 0.96]])
    targets = np.array([[0, 0, 0, 1],
                        [0, 0, 0, 1]])
    cross_entropy_loss = cross_entropy(predictions, targets)
    print("Cross entropy loss is: " + str(cross_entropy_loss))


if __name__ == '__main__':
    loss_test()
    print('finish!')

