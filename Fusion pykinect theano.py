import theano
import thread
import pytest
from pykinect import nui
from scipy import misc
import time
from theano import tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
np.set_printoptions(threshold='nan')

rng = RandomStreams()

"""a, b, c = misc.imread('a.png'), misc.imread('b.png'), misc.imread('c.png')
a, b, c = np.array(a, dtype=float)/255, np.array(b, dtype=float)/255, np.array(c, dtype=float)/255
a, b, c = np.reshape(a, (6636)), np.reshape(b, (6636)), np.reshape(c, (6636))
alfabet = np.zeros((3, 6636))
alfabet[0:,] = a
alfabet[1:,] = b
alfabet[2:,] = c

podpisy = [0, 1, 2]"""
ile_liter = 6

trening = [None] * (5*ile_liter)
for i in range(5*ile_liter):
    trening[i] = misc.imread('train/train'+str(i)+'.png')
    i += 1
trening = np.array(trening, dtype=float)/255
trening = np.reshape(trening, ((5*ile_liter), 6636))

podpisy = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5]

DEPTH_WINSIZE = 320, 240    # rozdzielczosc okna

screen_lock = thread.allocate()
screen = None

tmp_s = pytest.Surface(DEPTH_WINSIZE, 0, 16)    #(size, flagi, ile bitow)

generate = None     # zmienna do zapisywania obrazkow
handx, handy = 160, 120     # zmienna do wspolrzednych reki

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
        arr2d[190:270, 40:41] = 255
        arr2d[190:270, 125:126] = 255
        arr2d[190:191, 40:125] = 255
        arr2d[270:271, 40:125] = 255

        # Copy the depth map in the main surface
        pytest.surfarray.blit_array(screen, arr2d)

        # Update the screen
        pytest.display.update()


def skeleton_frame_ready(frame):
    skeletons = frame.SkeletonData
    global handx, handy
    for index, data in enumerate(skeletons):
        hand_p = data.SkeletonPositions[nui.JointId.hand_right]
        if hand_p.w == 1:
            handx = hand_p.x = (hand_p.x / hand_p.z)
            handy = hand_p.y = (hand_p.y / hand_p.z)
            handx = (int(630 * handx) + 320) / 2
            handy = (int(640 * handy) - 240) / (-2)

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

def model(X, h_w, o_w):
    hid = rectify(T.dot(X, h_w))
    out = T.nnet.softmax(T.dot(hid, o_w))
    return out


X = T.fmatrix()
Y = T.fmatrix()

h_w = init_weights((6636, 196))
o_w = init_weights((196, ile_liter))
#h_w.set_value(np.load('save1.npy'))
#o_w.set_value(np.load('save2.npy'))

py_x = model(X, h_w, o_w)
y_pred = T.argmax(py_x, axis=1)

w = [h_w, o_w]
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)

update = [[h_w, h_w - adagrad(gradient)[0] * 0.1],
          [o_w, o_w - adagrad(gradient)[1] * 0.1]]

answers = one_hot(podpisy, ile_liter)
print answers

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=py_x, allow_input_downcast=True)      # y_pred    + to cos ma wymiar 2D

#for i in range(300):
   # cost = train(trening, answers)
   # print cost

for i in range(300):
    cost = train(trening, answers)
    print cost

#np.save('save1.npy', h_w.get_value())
#np.save('save2.npy', o_w.get_value())

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
                #generate = pygame.surfarray.array2d(screen).T[(handy - 45):(handy + 45), (handx - 35):(handx + 35)]
                generate = pytest.surfarray.array2d(screen).T[41:125, 191:270]
                gen = np.array(generate, dtype=float) / 255
                gen = np.reshape(gen, (6636))
                test = np.zeros((1, 6636))
                test[0:, ] = gen
                for i in range(len(predict(test)[0])):
                    if predict(test)[(0,i)] > 0.999999:
                        print i, predict(test)[(0,i)]
                        time.sleep(0.2)

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