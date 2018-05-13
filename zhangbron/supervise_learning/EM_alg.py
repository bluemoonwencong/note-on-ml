# -*- coding:utf-8 -*-
from scipy import stats
import numpy as np
def em_single(priors, observations):
    """
        EM算法的单次迭代
        Arguments
        ------------
        priors:[theta_A,theta_B]
        theta_A: 在硬币A中theta为正面的概率
        observation:[m X n matrix]

        Returns
        ---------------
        new_priors:[new_theta_A,new_theta_B]
        :param priors:
        :param observations:
        :return:
    """

    counts = {'A':{'H':0,'T':0}, 'B':{'H':0,'T':0}}
    theta_A = priors[0]
    theta_B = priors[1]

    # E step
    for observation in observations:
        len_observation = len(observation)
        num_head = observation.sum()
        num_tail = len_observation - num_head

        # 二项分布
        contribution_A = stats.binom.pmf(num_head, len_observation, theta_A)
        contribution_B = stats.binom.pmf(num_head, len_observation, theta_B)

        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)
        # 更新在当前参数下A，B硬币产生的正反面次数
        counts['A']['H'] += weight_A * num_head
        counts['A']['T'] += weight_A * num_tail
        counts['B']['H'] += weight_B * num_head
        counts['B']['T'] += weight_B * num_tail

        # M step
    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A, new_theta_B]
        # contribution_A = scipy.stats.bino
        # p_A = 1.0
        # p_B = 1.0
        # for h_or_t in observation:
        #     if h_or_t == 1:
        #         p_A *= theta_A
        #         p_B *= theta_B
        #     else:
        #         p_A *= 1-theta_A
        #         p_B *= 1-theta_B
        # if p_A > p_B:
def em(observations,prior,tol = 1e-6,iterations=10000):
    """
    EM算法
    ：param observations :观测数据
    ：param prior：模型初值
    ：param tol：迭代结束阈值
    ：param iterations：最大迭代次数
    ：return：局部最优的模型参数
    """
    iteration = 0;
    while iteration < iterations:
        new_prior = em_single(prior,observations)
        delta_change = np.abs(prior[0]-new_prior[0])
        if delta_change < tol:
            break
        else:
            prior = new_prior
            iteration +=1
    return [new_prior,iteration]


if __name__ == '__main__':
    #
    observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                            [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                            [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                            [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
    result = em(observations,[0.6,0.5])
    print(result)