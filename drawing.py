import thread
import pytest
import scipy.misc
import numpy as np
from pykinect import nui


DEPTH_WINSIZE = 320, 240    # rozdzielczosc okna

screen_lock = thread.allocate()
screen = None

tmp_s = pytest.Surface(DEPTH_WINSIZE, 0, 16)    #(size, flagi, ile bitow)

generate = None     # zmienna do zapisywania obrazkow
#handx, handy = None, None     # zmienna do wspolrzednych reki
isPersonFound = False       # blokowanie odczytu wiecej niz 1 osoby
myIndex = None
temp_x1, temp_x2, temp_x3, temp_x4, temp_x5, counter1, counter2, lock = 0, 0, 0, 0, 0, 0, 0, False     # zmienne do funkcji analizy ruchu
temp_y1, temp_y2, temp_y3, temp_y4, temp_y5 = 0, 0, 0, 0, 0

pattern = np.zeros((240, 320))
pattern_list = []

class hand:
    x, y, z = 0, 0, 0

right = hand()
left = hand()

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

        # Copy the depth map in the main surface
        pytest.surfarray.blit_array(screen, arr2d)

        # Update the screen
        pytest.display.update()


def skeleton_frame_ready(frame):
    skeletons = frame.SkeletonData
    global right, left, isPersonFound, myIndex, counter1, counter2
    for index, data in enumerate(skeletons):
        hand_p = data.SkeletonPositions[nui.JointId.hand_right]
        hand_l = data.SkeletonPositions[nui.JointId.hand_left]

        if (hand_p.w == 1 and isPersonFound == False):
            isPersonFound = True
            myIndex = index

        if (isPersonFound and index == myIndex):
            right.x = hand_p.x = (hand_p.x/hand_p.z)
            #hand_l.x = (hand_l.x/hand_l.z)
            right.y = hand_p.y = (hand_p.y / hand_p.z)
            #hand_l.y = (hand_l.y / hand_l.z)
            #print hand_l.x, '\t', hand_p.x
            #print hand_l.y, '\t', hand_p.y
            right.x = (int(630 * right.x) + 320) / 2
            right.y = (int(640 * right.y) - 240) / (-2)
            matrix_of_changes(right)

    if counter1 == 4:
        start_analyze(right)
        counter1 = 0
    if counter2 == 7:
        stop_analyze(right)
        counter2 = 0
    counter1 += 1
    counter2 += 1

def start_analyze(right):
    global temp_x1, temp_x2, temp_x3, temp_y1, temp_y2, temp_y3, lock
    temp_x3 = temp_x2
    temp_x2 = temp_x1
    temp_x1 = right.x
    temp_y3 = temp_y2
    temp_y2 = temp_y1
    temp_y1 = right.y
    if (((21 > np.absolute((temp_x2 - temp_x1)) > 3) and (21 > np.absolute((temp_x3 - temp_x2)) > 3)) or ((21 > np.absolute((temp_y2 - temp_y1)) > 3) and (21 > np.absolute((temp_y3 - temp_y2)) > 3)))and lock == False:
        print 'Activated', temp_x3, temp_y3
        lock = True



def stop_analyze(right):
    global temp_x4, temp_x5, temp_y4, temp_y5, lock, pattern, pattern_list
    if (np.absolute((temp_x4 - right.x)) == 0) and (np.absolute((temp_x5 - right.x)) == 0) and (np.absolute((temp_y4 - right.y)) == 0) and (np.absolute((temp_y5 - right.y)) == 0) and lock == True:
        lock = False
        print "Deactivated", right.x
        pattern_list.append(pattern)
        for i in range(len(pattern_list)):
            scipy.misc.imsave('rysujemy/ksztalt' + str(i) + '.png', pattern_list[i])
        pattern = np.zeros((240, 320))
    temp_x4 = right.x
    temp_x5 = temp_x4
    temp_y4 = right.y
    temp_y5 = temp_y4

def matrix_of_changes(hand):
    global lock, pattern
    if lock == True:
        pattern[(hand.y - 2):(hand.y + 2), (hand.x - 2):(hand.x + 2)] = 255



def main():
    """Initialize and run the game."""
    pytest.init()

    # Initialize PyGame
    global screen, onoff
    screen = pytest.display.set_mode(DEPTH_WINSIZE, 0, 8)
    screen.set_palette(tuple([(i, i, i) for i in range(256)]))
    pytest.display.set_caption('PyKinect Depth Map')

    with nui.Runtime() as kinect:
        kinect.skeleton_engine.enabled = True
        kinect.depth_frame_ready += depth_frame_ready
        kinect.skeleton_frame_ready += skeleton_frame_ready
        kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution320x240,
                                 nui.ImageType.depth)


        # Main game loop
        while True:
            e = pytest.event.wait()
            global onoff



if __name__ == '__main__':
    main()