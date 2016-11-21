import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist

rng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def adagrad(gradient, bias=1e-6):
    ada = []
    for g in gradient:
        hist = 0
        gpow = g**2
        hist += (gpow.sum(axis=0)).sum(axis=0)
        ada.append(g / (bias + T.sqrt(hist)))
    return ada

def rectify(X):
    return T.maximum(X, 0.)

def dropout(X, prob=0.):
    if prob > 0:
        X *= rng.binomial(X.shape, n=1, p=prob, dtype=theano.config.floatX)
        X /= prob
    return X

def model(X, h_w, o_w):
    hid = T.nnet.sigmoid(T.dot(X, h_w))
    out = T.nnet.softmax(T.dot(hid, o_w))
    return out

trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

h_w = init_weights((784, 196))
o_w = init_weights((196, 10))

py_x = model(X, h_w, o_w)
y_pred = T.argmax(py_x, axis=1)

w = [h_w, o_w]
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)

update = [[h_w, h_w - adagrad(gradient)[0] * 0.1],
          [o_w, o_w - adagrad(gradient)[1] * 0.1]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
        # print trY[i]
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))