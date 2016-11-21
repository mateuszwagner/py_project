import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from scipy import misc
np.set_printoptions(threshold='nan')

rng = RandomStreams()
a, b, c = misc.imread('a.png'), misc.imread('b.png'), misc.imread('c.png')
a, b, c = np.array(a, dtype=float)/255, np.array(b, dtype=float)/255, np.array(c, dtype=float)/255
a, b, c = np.reshape(a, (6636)), np.reshape(b, (6636)), np.reshape(c, (6636))
alfabet = np.zeros((3, 6636))
alfabet[0:,] = a
alfabet[1:,] = b
alfabet[2:,] = c

podpisy = [0, 1, 2]

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

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

def model(X, o_w):
    out = T.nnet.softmax(T.dot(X, o_w))
    return out


X = T.fmatrix()
Y = T.fmatrix()

o_w = init_weights((6636, 3))

py_x = model(X, o_w)
y_pred = T.argmax(py_x, axis=1)

w = [o_w]
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)

update = [[o_w, o_w - adagrad(gradient)[0] * 0.1]]

answers = one_hot(podpisy, 3)
print answers

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(100):
    cost = train(alfabet, answers)
    print cost