# -*- coding:utf-8 -*-

# kNN(k-Nearest Neighbor)算法

# 导入科学计算包Numpy
from numpy import *
# 运算符模块，kNN执行排序会用到
import operator

# 创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
# k-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape是一个元组tuple (m,n) 取得行数m
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # tile重新复制堆叠成一个array，每一行都是inX的矩阵，每一行都减去dataSet这个向量
    sqDiffMat = diffMat ** 2 # 每个元素都平方
    sqDistances = sqDiffMat.sum(axis=1)  # 无参，所有相加得到标量；axis=0，每一列求和；axis=1，每一行求和，得到一个array(1,n)  即得到了距离的平方
    distances = sqDistances ** 0.5  # 开方
    sortedDistIndicies = distances.argsort() # 将distances从小到大排序后，提取其index下标构成的数组
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse = True)  # reverse 按照从大到小排序，返回list，选取得票最多的类别
    return sortedClassCount[0][0]