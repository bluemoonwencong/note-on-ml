# -*- coding:utf-8 -*-

import numpy as np
import pylab


def compute_error(b, m, data):
    totalError = 0

    x = data[:, 0]
    y = data[:, 1]
    totalError = (y - m * x - b) ** 2
    totalError = np.sum(totalError, axis=0)

    return totalError


def optimizer(data,starting_b,starting_m,learning_rate,num_iter):
    b = starting_b
    m = starting_m

    # gradient descent
    for i in range(num_iter):
        b, m = compute_gradient(b, m, learning_rate, data)
        if i%100 == 0:
            print('iter %d:error=%f' % (i, compute_error(b,m,data)))
    return [b, m]


def compute_gradient(b_current, m_current, learning_rate, data):
    b_gradient = 0
    m_gradient = 0

    N = float(len(data))

    # vector implementation
    x = data[:, 0]
    y = data[:, 1]
    b_gradient = -(2/N)*(y-m_current*x-b_current)
    b_gradient = np.sum(b_gradient,axis=0)
    m_gradient = -(2/N)*x*(y-m_current*x-b_current)
    m_gradient = np.sum(m_gradient,axis=0)

    # update m, b
    b_update = b_current - (learning_rate * b_gradient)
    m_update = m_current - (learning_rate * m_gradient)

    return [b_update, m_update]


def plot_data(data,b,m):

    #plottting
    x = data[:,0]
    y = data[:,1]
    y_predict = m*x+b
    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()


def linear_regression(training_data):
    # define learning rate
    # define y = mx + b
    learning_rate = 0.001
    init_b = 0.0
    init_m = 0.0
    number_iter = 1000

    # train model
    print('initial variables:\n init_b = %f\n init_m = %f\n error of begin = %f \n' % (init_b,init_m,compute_error(init_b,init_m,data)))

    # optimize b, m
    [b, m] = optimizer(data, init_b, init_m, learning_rate, number_iter)

    # print final b m error
    print('final formula parmaters:\n b = %f\n m=%f\n error of end = %f \n' % ( b, m, compute_error(b, m, data)))


    # plot result
    plot_data(data, b, m)


if __name__ == "__main__":

    data =np.loadtxt('./data/data.csv',delimiter=',')
    linear_regression(data)
    print(data.shape)