# -*- coding:utf-8 -*-
import numpy as np


def load_sim_data():
    data = np.matrix([[1. ,2.1],[2. , 1.1],[1.3 ,1.],[1. ,1.],[2. ,1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data, class_labels


def stump_classify(data_matrix, dimension, threshold_value, threshold_ineq):
    '''
        输入：数据矩阵，特征维数，某一特征的分类阈值，分类不等号
        功能：输出决策树桩标签
        输出：标签
    '''
    return_array = np.ones((np.shape(data_matrix)[0], 1))
    if threshold_ineq == 'lt':
        return_array[data_matrix[:, dimension] <= threshold_value] = -1
    else:
        return_array[data_matrix[:, dimension] >= threshold_value] = -1
    return return_array


def build_stump(data_array, class_labels, distibution):
    '''
        输入：数据矩阵，对应的真实类别标签，特征的权值分布
        功能：在数据集上，找到加权错误率（分类错误率）最小的单层决策树，显然，该指标函数与权重向量有密切关系
        输出：最佳树桩（特征，分类特征阈值，不等号方向），最小加权错误率，该权值向量D下的分类标签估计值
    '''
    data_matrix = np.mat(data_array)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_matrix)
    step_num = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf

    for i in range(n):
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max - range_min) / step_num
        for j in range(-1, int(step_num) + 1):
            for thresholdIneq in ['lt', 'gt']:
                threshold_value = range_min + float(j) * step_size
                predict_class = stump_classify(data_matrix, i, threshold_value, thresholdIneq)
                error_array = np.mat(np.ones((m, 1)))
                error_array[predict_class == label_mat] = 0
                weight_err = distibution.T * error_array
                if weight_err < min_error:
                    min_error = weight_err
                    best_class_est = predict_class.copy()
                    best_stump['dimen'] = i
                    best_stump['threshlod_value'] = threshold_value
                    best_stump['threshlod_ineq'] = thresholdIneq
    return best_class_est, min_error, best_stump


def add_boost_train_ds(data_arr, class_labels, num_iter=40):
    '''
        输入：数据集，标签向量，最大迭代次数
        功能：创建adaboost加法模型
        输出：多个弱分类器的数组
    '''
    weak_class = []
    m, n = np.shape(data_arr)
    D = np.mat(np.ones((m, 1))/m)
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_iter):
        best_class_est, min_err, best_stump = build_stump(data_arr, class_labels,D)
        print("D.T: ", end=" ")
        print(D.T)
        alpha = float(0.5 * np.log((1-min_err)/max(min_err, 1e-16)))
        print("alpha:", end=" ")
        print(alpha)
        best_stump['alpha'] = alpha
        weak_class.append(best_stump) #step3:将基本分类器添加到弱分类的数组中
        print("class_est: ", end=" ")
        print(best_class_est)

        expon = np.multiply(-1*alpha*np.mat(class_labels).T, best_class_est)

        D = np.multiply(D, np.exp(expon))
        D = D/D.sum() #step4:更新权重，该式是让D服从概率分布
        agg_class_est += alpha * best_class_est
        agg_class_est += alpha * agg_class_est  # steo5:更新累计类别估计值
        print("aggClassEst: ", end=" ")
        print(agg_class_est.T)
        print(np.sign(agg_class_est) != np.mat(class_labels).T)
        aggError = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        print("aggError: ", end=" ")
        print(aggError)
        aggErrorRate = aggError.sum() / m
        print("total error:", end="　")
        print(aggErrorRate)
        if aggErrorRate == 0.0:
            break
    return weak_class


def adaTestClassify(dataToClassify,weakClass):
    dataMatrix = np.mat(dataToClassify)
    m =np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(weakClass)):
        classEst = stump_classify(dataToClassify,weakClass[i]['dimen'],weakClass[i]['threshlod_value'],
                                 weakClass[i]['threshlod_ineq'])
        aggClassEst += weakClass[i]['alpha'] * classEst
        print(aggClassEst)
    return np.sign(aggClassEst)


if __name__  ==  '__main__':
    D =np.mat(np.ones((5,1))/5)
    dataMatrix ,classLabels= load_sim_data()
    bestClassEst, minError, bestStump = build_stump(dataMatrix,classLabels,D)
    weakClass = add_boost_train_ds(dataMatrix,classLabels,9)
    testClass = adaTestClassify(np.mat([0,0]),weakClass)


