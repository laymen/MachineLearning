# -*- coding: utf-8 -*-
from math import log

__author__ = 'Mouse'
from numpy import *

'''   创建一些实验样本，该函数返回的第一个变量是进行词条切分后文档集合 '''
'''    函数返回的第二个变量是一个类别标签的集合，这里有2类，侮辱和 正常言论。
这些文本是由人工标注 ，这些标志信息用于训练程序以便自动检测侮辱留言'''


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive 侮辱的文字, 0 not 正常的言论
    return postingList, classVec


''' 创建一个包含在所有文档中出现的不重复词的列表，为此使用Python的set数据类型，
    将词条输给set构造函数，set就会返回一个不重复词表'''


def createVocabList(dataSet):
    # 创建一个空集合，然后将每篇文档返回的新词集合添加到该集合中。
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 操作符|用于求两个集合的并集'''
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


''' 获得词汇表后，便可以使用函数setOfWord2Vec()，该函数的输入参数为词汇表及某个文档，输出的是文档向量，向量的每一个元素为1或0
    分别表示词汇表中的单词再输入文档中是否出现 '''

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个和词汇表等长的向量，并将其置于0
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1  # 遍历文档中的所以的单词，如果出现了词汇表中的单词，则将输出的文档向量中对应值设为1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec
#与上面唯一不同的是，每当遇到一个单词时，它会增加词向量中对应的值，而不是将对应的数值设为1
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 本次举例子采用的是6个
    numWords = len(trainMatrix[0])
    # 侮辱是数字1，正常言论是0,故 sum(trainCategory) 就是侮辱的言论
    # 侮辱的文档除以总的文档p(ci)
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 3/6=0.5
    p0Num = ones(numWords)
    p1Num = ones(numWords)  # change to ones()
    p0Denom = 0.0
    p1Denom = 0.0  # change to 2.0
                                   # 对每篇训练文档：
    for i in range(numTrainDocs):   # 对每个类别：
        if trainCategory[i] == 1:  # 如果词条出现在文档中-->增加该词条的计数值
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])  # 增加所有词条的计数值
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    print "文章类别属于1（将所有词条组的标签属于1的，相加起来如词条组2+词条组4+词条组6）："
    print p1Num
    print "属于p1Denom（词条组2中有8个1+词条组4中有5个1+词条组6中有6个1=19）: ", p1Denom
    print "文章类别属于0 (将所有词条组的标签属于0的，相加起来如词条组1+词条组3+词条组5）："
    print p0Num
    print "属于p0Denom（词条组1中有7个1+词条组3中有8个1+词条组5中有9个1=24）: ", p0Denom
    p1Vect = log(p1Num / p1Denom) # 将该词条的数目除以总词条数目得到的概率
    p0Vect = log(p0Num / p0Denom)  #
    return p0Vect, p1Vect, pAbusive  # 返回每个类别的条件概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #p(1)=
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult ln(ab)=ln(a)+ln(b)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ', classifyNB(thisDoc,p0V,p1V,pAb)

#---------------------------------------------------------------------------
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)

if __name__ == '__main__':
    # listOposts, listClasses = loadDataSet()
    # print "最原始的文档："
    # for i in range(len(listOposts)):
    #     print listOposts[i]
    # print "原始文档下的6个词条组分别对应的标签（1表示侮辱的话，0表示正常言论）：", listClasses
    # myVocabList = createVocabList(listOposts)
    # # 构建一个包含所有词列表
    # print "构建一个包含所有词列表:", "词汇个数为", len(myVocabList), ":"
    # print myVocabList
    # trainMat = []
    # for postinDoc in listOposts:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print "原始文档处理后的词条组："
    # for i in range(len(trainMat)):  # range(6) 表示，[0,1,2,3,4]
    #     print trainMat[i]
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print "返回每个类别的条件概率："
    # print "p0V:", p0V
    # print "p1V:", p1V
    # print "文档属于侮辱类的概率是：", pAb


    #testingNB()


    spamTest()


