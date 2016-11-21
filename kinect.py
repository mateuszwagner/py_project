import thread
import pygame
import numpy as np
from pykinect import nui
import scipy.misc
from scipy import ndimage
import time

DEPTH_WINSIZE = 320, 240    # rozdzielczosc okna

screen_lock = thread.allocate()
screen = None

tmp_s = pygame.Surface(DEPTH_WINSIZE, pygame.RESIZABLE, 16)    #(size, flagi, ile bitow)

generate = None     # zmienna do zapisywania obrazkow
gen_inc = 0
#handx, handy = None, None     # zmienna do wspolrzednych reki
isPersonFound = False       # blokowanie odczytu wiecej niz 1 osoby
myIndex = None
temp_x1, temp_x2, temp_x3, temp_x4, temp_x5, counter1, counter2, lock = 0, 0, 0, 0, 0, 0, 0, False     # zmienne do funkcji analizy ruchu
temp_y1, temp_y2, temp_y3 = 0, 0, 0

pattern = np.zeros((240, 320))

pygame.init()

class hand:
    x, y, z = 0, 0, 0

right = hand()
left = hand()

def depth_frame_ready(frame):
    with screen_lock:
        # Copy raw data in a temp surface
        frame.image.copy_bits(tmp_s._pixels_address)

        # Get actual depth data in mm
        arr2d = (pygame.surfarray.pixels2d(tmp_s) >> 3) & 4095  # przesuniecie bitowe o 3 (player index)

        # Process depth data as you prefer
        # arr2d = some_function(arr2d)
        # tutaj ma byc ta iteracja, ale sie nie da : ////////

        #       print arr2d[(250, 100)]
        arr2d += 255-2*arr2d    # odwrocenie tablicy, zalezne od bitow, wczesniej 256

        # Get an 8-bit depth map (useful to be drawn as a grayscale image)
        arr2d >>= 4
        """arr2d[190:271, 40:41] = 255
        arr2d[190:271, 121:122] = 255
        arr2d[190:191, 40:121] = 255
        arr2d[271:272, 40:121] = 255"""

        # Copy the depth map in the main surface
        pygame.surfarray.blit_array(screen, arr2d)

        # Update the screen
        pygame.display.update()


def skeleton_frame_ready(frame):
    skeletons = frame.SkeletonData
    global right, left, isPersonFound, myIndex, counter1, counter2
    for index, data in enumerate(skeletons):
        hand_p = data.SkeletonPositions[nui.JointId.hand_right]
        hand_l = data.SkeletonPositions[nui.JointId.hand_left]

        if (hand_p.w == 1 and isPersonFound == False):
            isPersonFound = True
            myIndex = index
        if (isPersonFound and index == myIndex and hand_p.z != 0):
            right.x = hand_p.x = (hand_p.x/hand_p.z)
            #hand_l.x = (hand_l.x/hand_l.z)
            right.y = hand_p.y = (hand_p.y / hand_p.z)
            #hand_l.y = (hand_l.y / hand_l.z)
            #print hand_l.x, '\t', hand_p.x
            #print hand_l.y, '\t', hand_p.y
            right.x = (int(630 * right.x) + 320) / 2
            right.y = (int(640 * right.y) - 240) / (-2)
            matrix_of_changes(right)
            #print right.x
    #start_analyze(right)
    #stop_analyze(right)
    if counter1 == 4:
        start_analyze(right)
        #print right.x, right.y
        counter1 = 0
    if counter2 == 5:
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
    global temp_x4, temp_x5, lock
    if (np.absolute((temp_x4 - right.x)) == 0) and (np.absolute((temp_x5 - right.x)) == 0) and lock == True:
        lock = False
        print "Deactivated", right.x
    temp_x4 = right.x
    temp_x5 = temp_x4


def matrix_of_changes(hand):
    global lock, pattern
    if lock == True:
        pattern[(hand.y - 2):(hand.y + 2), (hand.x - 2):(hand.x + 2)] = 255
    if lock == False:
        scipy.misc.imsave('rysujemy/ksztalt.png', pattern)



def main():
    """Initialize and run the game."""
   # pygame.init()
    gameExit = False

    # Initialize PyGame
    global screen
    screen = pygame.display.set_mode(DEPTH_WINSIZE, pygame.RESIZABLE, 8)
    screen.blit(pygame.image.load('ryjek.png'), (30,30))

    screen.set_palette(tuple([(i, i, i) for i in range(256)]))
    pygame.display.set_caption('PyKinect Depth Map')

    with nui.Runtime() as kinect:
        kinect.skeleton_engine.enabled = True
        kinect.depth_frame_ready += depth_frame_ready
        kinect.skeleton_frame_ready += skeleton_frame_ready
        kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution320x240,
                                 nui.ImageType.depth)


        # Main game loop
        while not gameExit:
            e = pygame.event.wait()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_c:
                    global gen_inc
                    generate = pygame.surfarray.array2d(screen).T[(right.y - 30):(right.y + 25), (right.x - 40):(right.x + 15)]
                    for x in range(len(generate)):
                        for i in range(len(generate)):
                            if generate[x, i] < 150:
                                generate[x, i] = 0
                    #generate = pygame.surfarray.array2d(screen).T[41:121, 191:271]
                    #sx = ndimage.sobel(generate, axis=0, mode='constant')
                    #sy = ndimage.sobel(generate, axis=1, mode='constant')
                    #sob = np.hypot(sx, sy)
                    #sob = (sob+generate)/2
                    scipy.misc.imsave('train/train'+str(gen_inc)+'.png', generate)
                    gen_inc += 1
                    print gen_inc

                if e.key == pygame.K_v:
                    gen_inc -= 1
                    print gen_inc

            elif e.type == pygame.QUIT:
                gameExit = True
                break


if __name__ == '__main__':
    main()