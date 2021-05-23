import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_grad(s):
    return s * (1.0 - s)

def relu(x):
    return x * (x > 0)

def  relu_grad(x):
    return 1.0 * (x > 0)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_grad(t):
    return 1 - t**2


#with numerical stability
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def logloss(x, y):
    probs = softmax(x)
    return probs, -y * np.log(probs)

def logloss_grad(probs, y):
    probs[:,y] -= 1.0
    return probs

def MSE(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def MSE_grad(y_true, y_pred):
    # print(x.shape)
    # print(y_true.shape)
    # print(y_pred.shape)
    # return -((y_true - y_pred).dot(x)).mean()
    # return 2 * (np.dot(x.T, (y_pred - y_true))).mean()
    return 2.0 * (y_true - y_pred).mean()

def batch_hits(x, y):
    return np.sum(np.argmax(x, axis=1) == y)
