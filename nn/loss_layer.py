from nn.funcs import *
from nn.op import *
import numpy as np

#implements a log loss layer
class loss_layer(op):

    def __init__(self, i_size, o_size, loss_func, loss_func_grad, is_regression):
        super(loss_layer, self).__init__(i_size, o_size)
        self.grads = np.zeros((o_size, i_size))
        self.loss_func = loss_func
        self.loss_func_grad = loss_func_grad
        self.is_regression = is_regression

    def forward(self, x):
        self.x = x
        self.o = np.dot(x, self.W) + self.b
        if self.is_regression:
            # return self.o.flatten()
            return self.o
        return softmax(self.o)
        

    #alpha is used as reward in some reinforcement learning envs
    def backward(self, y, rewards=None):
        if self.is_regression:
            self.grads = self.loss_func_grad(y.reshape(-1,1), self.o)
        else:
            one_hot = np.zeros(self.o.shape)
            one_hot[np.arange(self.o.shape[0]), y] = 1
            if rewards is not None:
                self.grads = (one_hot - self.o) * rewards
            else:
                self.grads = one_hot - self.o

    def loss(self, y):
        if self.is_regression:
            return self.loss_func(y.reshape(-1,1), self.o)
        one_hot = np.zeros(self.o.shape, dtype=np.int)
        one_hot[np.arange(self.o.shape[0]), y] = 1
        #fixed_section = np.nan_to_num((1 - one_hot) * np.log(1 - self.o))
        return -np.mean(np.sum(one_hot * np.log(self.o + 1e-15), axis=1))

    def update(self, lr):
        if self.is_regression:
            print(self.grads)
            self.W += lr * self.grads
            self.b += lr * self.grads
        else:
            self.W += lr * np.dot(self.x.T, self.grads)
            self.b += lr * np.mean(self.grads, axis = 0)


