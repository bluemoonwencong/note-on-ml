# -*- coding:utf-8 -*-

import numpy as np


class NaiveBayes:
    def __init__(self):
        self.creteria = "NB"

    def createVocList(self, dataList):
        """创建一个词库向量"""
        voccabSet = set([])
        for line in dataList:
            print(set(line))
            voccabSet = voccabSet | set(line)
        return list(voccabSet)

    def setOfWordtoVec(self, vocablist, inputSet):
        """
        功能：根据给定的一行词，将每个词映射到词库向量中，出现则标记1，不出现则为零
        """
        outPutVec = [0] * len(vocablist)
        for word in inputSet:
            if word in vocablist:
                outPutVec[vocablist.index(word)] = 1
            else:
                print('the word: %s is not in the vocabulary' % word)
        return outPutVec

    def bagOfWordtoVecMN(self, vocablist, input):
        """
        对每行词使用第二种统计策略，统计单词个数，然后映射到词库中
        :param vocablist:
        :param input:
        :return: 一个n维向量，n为词库长度，每个取值为单词出现次数
        """
        output = [0] * len(vocablist)
        for word in input:
            if word in vocablist:
                output[vocablist.index(word)] += 1
        return output

    def train(self, trainMatrix, trainlabels):
        """
        :param trainMatrix:
        :param trainlabels:
        :return: 计算条件概率和类别标签
        """
        numTrainDoc = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pNegitive = sum(trainlabels) / float(numTrainDoc) # 负样本出现概率

        p0Num = np.ones(numWords) # 初始样本个数为1，防止条件概率为0
        p1Num = np.ones(numWords)

        p0InAll = 2.0 # 使用拉普拉斯平滑
        p1InAll = 2.0

        # 在单个文档和整个词库中更新正负样本数
        for i in range(numTrainDoc):
            if trainlabels[i] == 1:
                p1Num += trainMatrix[i]
                p1InAll += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0InAll += sum(trainMatrix[i])

        print(p1InAll)

        # 计算给定类别的条件下，词汇表中单词出现的概率
        # 然后取log对数，解决条件概率乘积下溢
        p0Vect = np.log(p0Num/p0InAll) #计算类标签为0时的其它属性发生的条件概率
        p1Vect = np.log(p1Num/p1InAll)

        return p0Vect, p1Vect, pNegitive

    def classifyNB(self, vec_sample, p0Vec, p1Vec, pNeg):
        """
        使用朴素贝叶斯进行分类
        :param vec_sample:
        :param p0Vec:
        :param p1Vec:
        :param pNeg:
        :return: 返回0或者1
        """
        prob_y0 = sum(vec_sample * p0Vec) + np.log(1-pNeg)
        prob_y1 = sum(vec_sample * p1Vec) + np.log(pNeg)
        if prob_y0 < prob_y1:
            return 1
        else:
            return 0

    def testNB(self, sample):
        listpost, listClass = loadDataSet()
        my_vocabulary_list = self.createVocList(listpost)

        train_mat = []
        for post_doc in listpost:
            train_mat.append(self.bagOfWordtoVecMN(my_vocabulary_list, post_doc))
        p0V, p1V, pAb = self.train(np.array(train_mat), np.array(listClass))
        print(train_mat)

        test_sample = np.array(self.bagOfWordtoVecMN(my_vocabulary_list, sample))
        result = self.classifyNB(test_sample, p0V, p1V, pAb)
        print('test sample result is:')
        print(result)
        return result

########## 加载训练数据
def loadDataSet():
    wordsList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', ' and', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classLable = [0, 1, 0, 1, 0, 1]  # 0：good; 1:bad
    return wordsList, classLable


if __name__ == "__main__":
    clf = NaiveBayes()
    testEntry = [['love', 'my', 'girl', 'friend'],
                 ['stupid', 'garbage'],
                 ['Haha', 'I', 'really', "Love", "You"],
                 ['This', 'is', "my", "dog"]]
    for i in range(len(testEntry)):
        print("第 %d 条的分类结果：" % i)
        clf.testNB(testEntry[i])
        print()