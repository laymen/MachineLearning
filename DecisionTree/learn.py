# -*- coding: utf-8 -*-
__author__ = 'Mouse'
from math import log
import operator
import treePlotter


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    #print "numEntries:", numEntries
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]  #取dataSet最后的一列数据
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #print " labelCounts:", labelCounts  # {'yes': 2, 'no': 3}
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  #如 yes: 2/5=0.4  如no :3/5=0.6
        #print key, ":", prob
        shannonEnt -= prob * log(prob, 2)  #log base 2
        #print key, ":", shannonEnt
    ''''此处可以设置信息增益的最小值，信息增益小就不分裂了,可以一定程度控制过拟合 '''
    return shannonEnt


# dataSet是待划分的数据集、划分数据集的特征、特征的返回值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 取出每行的第一个元素进行比较
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    print "baseEntropy:", baseEntropy
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  #create a list of all the examples of this feature
        uniqueVals = set(featList)  #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  #compare this to the best gain so far
            bestInfoGain = infoGain  #if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


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
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 使用决策树执行分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    if firstStr in featLabels:
        featIndex = featLabels.index(firstStr)
        for key in secondDict.keys():
            if testVec[featIndex] == key:  #比较特征值，决策树是根据特征的值划分的
                if type(secondDict[key]).__name__ == 'dict':  #比较是否到达叶结点
                    classLabel = classify(secondDict[key], featLabels, testVec)  #递归调用
                else:
                    classLabel = secondDict[key]
        return classLabel


def storeTree(inputTree, filename):
    import pickle

    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle

    fr = open(filename)
    return pickle.load(fr)


def deal():
    lenses = []
    with open("lenses.txt") as file:
        for line in file:
            tokens = line.strip().split('\t')
            lenses.append([tk for tk in tokens])
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)

if __name__ == '__main__':
    deal()
    # dataSet, labels = createDataSet()
    # print "用于训练决策树的原始数据dataSet：", dataSet
    # print "用于训练的标签labels：", labels

    # shannonEnt = calcShannonEnt(dataSet)
    # print shannonEnt
    # retDataSet = splitDataSet(dataSet, 0, 0)
    # print retDataSet
    # bestFeature = chooseBestFeatureToSplit(dataSet)
    # print bestFeature

    # myTree = createTree(dataSet, labels)
    # storeTree(myTree, 'classifierStorage.txt')
    # a = grabTree('classifierStorage.txt')
    # print "a:", a
    # print "训练完成后，生成的决策树myTree:", myTree

    # treePlotter.createPlot()
    # myTree = treePlotter.retrieveTree(1)

    # treePlotter.createPlot(myTree)
    # dataSetTest, labelsTest = createDataSet()
    # classLabel = classify(myTree, labelsTest, [1, 1])
    # print "测试后classLabel:", classLabel



