import numpy as np
import cv2
import random
import math
from matplotlib import pyplot as plt
from Network import Network


import sys
sys.setrecursionlimit(1000000)


def sp_noise(image, proc):
    needed_amount_noises = int(image.shape[0] * image.shape[1] * proc)
    amount_noises = 0
    output = np.zeros(image.shape,np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < proc and amount_noises <= needed_amount_noises:
                amount_noises += 1
                if image[i, j] == 255:
                    output[i][j] = 0
                else:
                    output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == '__main__':
    image_1_path = 'Images/lab4_1.png'
    image_2_path = 'Images/lab4_2.png'
    image_3_path = 'Images/lab4_3.png'

    image_1 = cv2.imread(image_1_path, cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread(image_2_path, cv2.IMREAD_GRAYSCALE)
    image_3 = cv2.imread(image_3_path, cv2.IMREAD_GRAYSCALE)

    net = Network(3, [36, 26, 3])

    # net.add_learn_image(image_1, [1, 0, 0])
    net.add_learn_image(image_2, [0, 1, 0])
    net.add_learn_image(image_3, [0, 0, 1])

    net.learn()

    noisy_percent = 0
    noisy_image_t = sp_noise(image_1, noisy_percent)
    noisy_image_l = sp_noise(image_2, noisy_percent)
    noisy_image_v = sp_noise(image_3, noisy_percent)

    # net.add_play_image(noisy_image_t)
    # net.add_play_image(noisy_image_l)
    # net.add_play_image(noisy_image_v)
    #
    # output = net.play_network()

    # cv2.imshow('filtered', noisy_image_t)
    # cv2.imshow('filtered_image', output[0])
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # titles = ['Noisy T', 'Noisy L', 'Noisy V', 'Letter T', 'Letter L', 'Letter V']
    # images = [noisy_image_t, noisy_image_l, noisy_image_v]
    #
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    #
    # plt.show()