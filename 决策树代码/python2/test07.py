# encoding: utf-8
from math import log
import operator
import treePlotter


# 获取数据
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


dataSet, labels = createDataSet()



# 计算系统熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 获取数组的长度
    labelCounts = {}
    for featVec in dataSet:  # 计算数据集中不同类别的个数
        currentLabel = featVec[-1]  # 获取类别
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 概率
        shannonEnt -= prob * log(prob, 2)  # 熵
    return shannonEnt


shannonEnt = calcShannonEnt(dataSet)


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet




def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)  # 系统熵
    bestInfoGain = 0.0;
    bestFeature = -1  # 初始最好的信息增益及特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  # 取出特征值
        uniqueVals = set(featList)  # 防止重复
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # 信息增益
        if (infoGain > bestInfoGain):  # 计算最大信息增益
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    print bestFeatLabel
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree



# fr = open("lenses.txt")
# lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# lensesLabel = ['age','prescript','astigmatic','tearRate']
# print splitDataSet(lenses,4,set(lenses))
# myTree = createTree(lenses, lensesLabel)
myTree = createTree(dataSet, labels)

treePlotter.createPlot(myTree)
# print splitDataSet(lenses, 0, "young")