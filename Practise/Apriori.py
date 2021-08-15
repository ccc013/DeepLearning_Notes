#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/5 15:38
@Author  : luocai
@file    : Apriori.py
@concat  : 429546420@qq.com
@site    : 
@software: PyCharm Community Edition 
@desc    :

"""


def loadDataSet():
    # a simple test dataset
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 创建一个大小为 1 的所有候选项集的集合
def createC1(dataset):
    C1 = []
    for transaction in dataset:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    # 排序
    C1.sort()
    # python3 使用 map 直接返回的是一个对象
    return list(map(frozenset, C1))


# 计算所有项集的支持度
def scanD(D, Ck, minSupport):
    '''
    对输入的候选项集 Ck，计算每个项集的支持度，并返回符合要求的项集和所有频繁项集
    :param D: 数据集
    :param Ck: 包括 k 个物品的候选项集
    :param minSupport: 最小支持度
    :return:
        retList: 满足最小支持度的候选项集
        supportData: 所有的频繁项集和其支持度的数据
    '''
    # ssCnt 是存储每个候选项集的支持度的字典
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ssCnt.setdefault(can, 1)
                ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}
    # 计算支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support

    return retList, supportData


def aprioriGen(Lk, k):
    '''
    生成 Ck 候选项集
    :param Lk: 频繁项集列表
    :param k: 项集元素个数
    :return: 返回指定物品数量的候选项集
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            # 获取两个相邻项集的前 k-2 项
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                # 如果两个集合相同，进行合并
                retList.append(Lk[i] | Lk[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    '''
    Apriori 算法入口
    :param dataSet: 数据集
    :param minSupport:  最小支持度
    :return:
        L: 最终的满足最小支持度的候选项集
        supportData: 所有的频繁项集和其支持度的数据
    '''
    C1 = createC1(dataSet)
    # 创建集合表示的数据集
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k - 2]) > 0:
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        # 增加候选项集的物品数量
        k += 1
    return L, supportData

def rulesFromConseq(freqSet, H, supportData, br1, minConf=0.7):
    '''
    生成候选规则集合
    :param freqSet: 频繁项集
    :param H: 规则右部的元素列表
    :param supportData:
    :param br1:
    :param minConf: 可信度阈值
    :return:
    '''
    m = len(H[0])




if __name__ == '__main__':
    dataSet = loadDataSet()
    print('test dataset:', dataSet)
    # C1 = createC1(dataSet)
    # print('C1:', C1)
    # # 构建集合表示的数据集 D
    # D = list(map(set, dataSet))
    # print('D:', D)
    # L1, suppData0 = scanD(D, C1, minSupport=0.5)
    # print('L1:', L1)

    L, suppData = apriori(dataSet)
    for i, val in enumerate(L):
        print('{}: {}'.format(i, val))
    for key, data in suppData.items():
        print('{}: {}'.format(key, data))
