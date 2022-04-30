import cv2
import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from math import pi


def print_IDs():
    print("205896103 , 211406343\n")


def Threshold(im, thres):
    thres_im = np.zeros(im.shape)
    thres_im[im[:] > thres] = 255
    return thres_im


def sobel_edging(im):
    aux_vec = np.array([1, 2, 1])
    Sx = np.zeros((3, 3))
    Sy = np.zeros((3, 3))

    Sx[0, :] = aux_vec[:]
    Sx[2, :] = -1 * aux_vec[:]

    Sy[:, 0] = aux_vec[:]
    Sy[:, 2] = -1 * aux_vec[:]

    Sx = Sx / 2
    Sy = Sy / 2

    df_dxH = 2 * convolve2d(im, Sx)
    df_dyH = 3 * convolve2d(im, Sy)

    mag = np.sqrt(np.square(df_dxH) + np.square(df_dyH))
    mag = Threshold(mag, 125)

    return np.uint8(mag)


def CannyThreshold(im, low_val, high_val, kernel_size):
    img_blur = cv2.cv2.blur(im, (3, 3))
    detected_edges = cv2.cv2.Canny(img_blur, low_val, high_val, kernel_size)
    mask = detected_edges != 0
    dst = im * (mask[:, :].astype(im.dtype))
    return dst


def ShowLines(im, low_val, high_val):
    black_im = np.zeros(im.shape, dtype="uint8")
    im_edges = cv2.Canny(im, low_val, high_val)
    im_lines = cv2.HoughLinesP(im_edges, 1, pi / 180, 100, minLineLength=10, maxLineGap=250)

    for line in im_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(black_im, (x1, y1), (x2, y2), (255, 0, 0), 3)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(black_im, cmap='gray')
