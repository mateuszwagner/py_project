import os
os.environ["THEANO_FLAGS"] = "floatX=float32"
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import mnist
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.pool import pool_2d
import thread
import pytest
from pykinect import nui
from scipy import misc
import time

srng = RandomStreams()

ile_liter = 6

trening = [None] * (5*ile_liter)
for i in range(5*ile_liter):
    trening[i] = misc.imread('train_skeletonrdy/train'+str(i)+'.png')
    i += 1
trening = np.array(trening, dtype=float)/255
trening = np.reshape(trening, ((5*ile_liter), 1, 90, 70))
print trening.shape

podpisy = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]

DEPTH_WINSIZE = 320, 240    # rozdzielczosc okna

screen_lock = thread.allocate()
screen = None

tmp_s = pytest.Surface(DEPTH_WINSIZE, 0, 16)    #(size, flagi, ile bitow)

generate = None     # zmienna do zapisywania obrazkow
hand = None     # zmienna do wspolrzednych reki

def depth_frame_ready(frame):
    with screen_lock:
        # Copy raw data in a temp surface
        frame.image.copy_bits(tmp_s._pixels_address)

        # Get actual depth data in mm
        arr2d = (pytest.surfarray.pixels2d(tmp_s) >> 3) & 4095  # przesuniecie bitowe o 3 (player index=3)

        # Process depth data as you prefer
        # arr2d = some_function(arr2d)
        arr2d += 256-2*arr2d    # odwrocenie tablicy

        # Get an 8-bit depth map (useful to be drawn as a grayscale image)
        arr2d >>= 4
        arr2d[190:271, 40:41] = 255
        arr2d[190:271, 121:122] = 255
        arr2d[190:191, 40:121] = 255
        arr2d[271:272, 40:121] = 255

        # Copy the depth map in the main surface
        pytest.surfarray.blit_array(screen, arr2d)

        # Update the screen
        pytest.display.update()


def skeleton_frame_ready(frame):
    skeletons = frame.SkeletonData
    global hand
    for index, data in enumerate(skeletons):
        hand_pos = data.SkeletonPositions[nui.JointId.hand_right]
        if hand_pos.w == 1:
            hand = hand_pos
            print hand_pos

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

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

#trX, teX, trY, teY = mnist(onehot=True)

#trX = trX.reshape(-1, 1, 28, 28)
#teX = teX.reshape(-1, 1, 28, 28)

#print trX.shape

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 9 * 10, 625))
w_o = init_weights((625, 6))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

answers = one_hot(podpisy, ile_liter)
print answers

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(10):
    cost = train(trening, answers)
    print cost

def main():
    """Initialize and run the game."""
    pytest.init()

    # Initialize PyGame
    global screen
    screen = pytest.display.set_mode(DEPTH_WINSIZE, 0, 8)
    screen.set_palette(tuple([(i, i, i) for i in range(256)]))
    pytest.display.set_caption('PyKinect Depth Map')

    with nui.Runtime() as kinect:
        #kinect.skeleton_engine.enabled = True
        kinect.depth_frame_ready += depth_frame_ready
        #kinect.skeleton_frame_ready += skeleton_frame_ready
        kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution320x240,
                                 nui.ImageType.depth)


        # Main game loop
        while True:
            e = pytest.event.wait()

            while True:
                generate = pytest.surfarray.array2d(screen).T[41:121, 191:271]
                gen = np.array(generate, dtype=float) / 255
                gen = np.reshape(gen, (1, 90, 70))
                test = np.zeros((1, 1, 90, 70))
                test[0:, ] = gen
                print predict(test)
                time.sleep(1)

            if e.type == pytest.KEYDOWN:
                if e.key == pytest.K_c:
                    """generate = pygame.surfarray.array2d(screen).T[41:125, 191:270]
                    gen = np.array(generate, dtype=float) / 255
                    gen = np.reshape(gen, (6636))
                    test = np.zeros((1, 6636))
                    test[0:, ] = gen
                    print predict(test)"""

            elif e.type == pytest.QUIT:
                break


if __name__ == '__main__':
    main()