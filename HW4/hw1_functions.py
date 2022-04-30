import cv2.cv2 as cv2

import numpy as np
import matplotlib.pyplot as plt


def print_IDs():
    print("205896103 , 211406343\n")


def contrastEnhance(im, gray_range):
    """

    :param im: A grayscale image in the range [0, 255].

    :param gray_range: Range of gray values in the form [minValue, maxValue].

    :return: New image indicating `im` with the gray values being in gray_range.

    :raise ValueError: if `gray_range` is not a 2 - sized tuple or if the values in it are invalid.

    """
    if len(gray_range) != 2 or gray_range[1] < gray_range[0] or gray_range[0] < 0 or gray_range[1] > 255:
        raise ValueError("Invalid range, expected a range of [minVal, maxVal] where 0 <= minVal <= maxVal <= 255")
    im_max = im.max()
    im_min = im.min()
    b = gray_range[0]
    a = (gray_range[1]-gray_range[0])/(im_max-im_min)
    nim = (a*(im-im_min)).round().astype(int)+b
    return nim, a, b


def showMapping(old_range, a, b):
    imMin = np.min(old_range)
    imMax = np.max(old_range)
    x = np.arange(imMin, imMax+1, dtype=np.float)
    y = a * x + b
    plt.figure()
    plt.plot(x, y)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.title('contrast enhance mapping')


def minkowski2Dist(im1, im2):
    """Returns Minkowski distance between 2 image histograms: `im1` and `im2` ."""
    im1_hist = np.histogram(im1, bins=256, range=(0,255))[0]
    im2_hist = np.histogram(im2, bins=256, range=(0,255))[0]
    return int(pow(sum(pow(abs(im1_hist-im2_hist), 2)), 0.5))


def meanSqrDist(im1, im2):
    """

    :param im1: First matrix of grayscale image in the range [0, 255].

    :param im2: Second matrix of grayscale image in the range [0, 255], and it has the same size as im1.

    :return: Mean square distance between im1 and im2.

    :raise ValueError: if the two received matrices im1 and im2 are not of the same size.

    """
    if im1.size == 0:   # avoiding dividing by 0
        return 0
    if im1.size != im2.size:
        raise ValueError("Expected matrices of the same size.")
    im_diff = np.sum(pow(im1-im2, 2))/im1.size
    return im_diff


def sliceMat(im):
    """

    :param im: A 2D grayscale image matrix in the range [0, 255].

    :return: The Slice Matrix, a binary valued matrix of size numPixel X 256, where numPixel is the number of pixels
     in the received image im1.

    """
    SL = np.zeros((im.size, 256), dtype=int)
    rows, cols = im.shape
    for i in range(rows):
        for j in range(cols):
            SL[cols*i+j][im[i][j]] = 1
    return SL


def SLTmap(im1, im2):
    """

    :param im1: Grayscale image in the range [0, 255].

    :param im2: Grayscale image int the range [0, 255] with the same size as im1.

    :return: The tuple (nim, TM) where nim is a new grayscale image that indicates im1 after a tone mapping
    making it look similar to im2 as possible, and TM is the tone map used to get nim .

    :raise ValueError: if the two matrices received im1 and im2 are of different size.

    """
    if np.size(im1) != np.size(im1):
        raise ValueError("Expected images of the same size.")

    im1_SL = sliceMat(im1)
    im1_r, im1_c = im1.shape
    rows, cols = im1_SL.shape
    TM = np.zeros(cols, dtype=int)
    for col in range(cols):
        grey_avrg = 0
        count = 0
        for row in range(rows):
            if im1_SL[row][col] == 1:
                grey_avrg = grey_avrg + im2[int((row - row % im1_c) / im1_c)][int(row % im1_c)]
                count = count + 1
        if grey_avrg != 0:
            grey_avrg = grey_avrg / count
        TM[col] = int(grey_avrg)
    return mapImage(im1, TM), TM


def mapImage(im, tm):
    """

    :param im: Grayscale image in the range [0, 255].

    :param tm: Tone map, a 1x256 numpy array defining a tone.

    :return: Maps im1 grayscale according to tm and returns the results without changing the values of im itself.

    """
    TMim = np.zeros(im.shape, dtype=int)
    SL = sliceMat(im)
    rows, cols = SL.shape
    im_r, im_c = im.shape
    TMim = np.matmul(SL, tm).reshape(np.shape(im))
    return np.uint8(TMim)


def sltNegative(im):
    """Returns the negative image of the received image `im1`."""
    return mapImage(im, np.arange(255, -1, -1, dtype='int'))


def sltThreshold(im, thresh):
    """

    :param im: Grayscale image matrix in the range [0, 255].

    :param thresh: The thresh value.

    :return: Grayscale image containing the content of im1 but after doing a threshold for the it according to the value
    of the parameter thresh.

    """
    MT = np.zeros(256, dtype=int)
    MT[thresh+1:] = 255
    return mapImage(im, MT)
