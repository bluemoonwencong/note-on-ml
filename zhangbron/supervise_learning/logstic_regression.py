# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class LR:
    def __init__(self):
        self.dim = 2
        self.w = np.array([1.0, 1.0])
        self.b = 0
        self.eta = 0.2

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def logistic_regression(self, x, y, eta):
        itr = 0
        row, column = np.shape(x)
        xpts = np.linspace(-1.5, 2.5)
        while itr<1000:
            fx = np.dot(self.w, x.T) + self.b
            hx = self.sigmoid(fx)
            t = (hx - y)
            s = [[i[0] * i[1][0], i[0] * i[1][1]] for i in zip(t, x)]
            gradient_w = np.sum(s, 0) / row * self.eta
            gradient_b = np.sum(t, 0) / row * self.eta
            self.w -= gradient_w
            self.b -= gradient_b
            ypts = (self.w[0] * xpts + self.b) / (-self.w[1])
            if itr % 100 == 0:
                plt.figure()
                for i in range(350):
                    plt.plot(x[i, 0], x[i, 1], col[y[i]] + 'o')
                plt.ylim([-1.5, 1.5])
                plt.plot(xpts, ypts, 'g-', lw=2)
                plt.title('eta = %s, Iteration = %s\n' % (str(eta), str(itr)))
                plt.savefig('p_N%s_it%s' % (str(row), str(itr)), dpi=200, bbox_inches='tight')
            itr += 1


if __name__ == '__main__':

    x, y = make_moons(350, noise=0.0)
    col = {0: 'r', 1: 'b'}
    lr = LR()
    lr.logistic_regression(x, y, eta=1.2)
    plt.show()


