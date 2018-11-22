import numpy as np
import cv2
import random
import math
from matplotlib import pyplot as plt
from Network import Network


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
    image_patches = []
    for i in range(1, 6):
        image_patches.append('Images/lab4_' + str(i) + '.png')

    images = []
    for i in range(0, 5):
        images.append(cv2.imread(image_patches[i], cv2.IMREAD_GRAYSCALE))

    net = Network(3, [36, 26, 5], learning_rate=0.1, max_error=0.005)

    net.add_learn_image(images[0], [1, 0, 0, 0, 0])
    net.add_learn_image(images[1], [0, 1, 0, 0, 0])
    net.add_learn_image(images[2], [0, 0, 1, 0, 0])
    net.add_learn_image(images[3], [0, 0, 0, 1, 0])
    net.add_learn_image(images[4], [0, 0, 0, 0, 1])

    net.learn()

    noisy_percent = 0
    noisy_images = []
    for i in range(0, 5):
        noisy_images.append(sp_noise(images[i], noisy_percent))

    results = []
    for i in range(0, 5):
        results.append(str(net.play_image(noisy_images[i])))

    # cv2.imshow('filtered', noisy_image_t)
    # cv2.imshow('filtered_image', output[0])
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(5):
        plt.subplot(5, 1, i + 1), plt.imshow(noisy_images[i], 'gray')
        plt.title(results[i])
        plt.xticks([]), plt.yticks([])

    plt.show()