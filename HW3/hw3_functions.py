import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def addSPnoise(im, p):
    sp_noise_im = im.copy()
    sp_noise_im = np.ravel(sp_noise_im)
    num_of_bad_pts = p * sp_noise_im.size
    bad_points_indices = random.sample(range(sp_noise_im.size), int(num_of_bad_pts))
    dark_indices_in_bad_pts_list = list(random.sample(range(int(sp_noise_im.size)), int( num_of_bad_pts/2 )))
    sp_noise_im[bad_points_indices] = 255
    sp_noise_im[dark_indices_in_bad_pts_list] = 0
    sp_noise_im = np.reshape(sp_noise_im,im.shape)
    return sp_noise_im


def addGaussianNoise(im, s):
    gaussian_noise_im = im.copy()
    gauss_noise = np.random.normal(0, s, im.shape)
    gauss_noise = gauss_noise.reshape(im.shape)
    gaussian_noise_im = gaussian_noise_im + gauss_noise
    return np.uint8(gaussian_noise_im)


def cleanImageMedian(im, radius):
    median_im = im.copy()

    for row in range(radius, median_im.shape[0] - radius ):
        for col in range(radius, median_im.shape[1] - radius ):
            window = im[row-int(radius/2):row+int(radius/2)+1,col-int(radius/2):col+int(radius/2)+1]
            median = np.uint8(np.median(window))
            median_im[row, col] = median

    return median_im

def gkern(l, sig):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def cleanImageMean(im, radius, maskSTD):
   	# TODO: add implementation
    filter = gkern(radius,maskSTD)
    cleaned_im = convolve2d(im,filter,mode='same')
    return cleaned_im


def bilateralFilt(im, radius, stdSpatial, stdIntensity):
    bilateral_im = im.copy()
    dist_xj = np.zeros((radius,radius))
    center = (radius - 1) / 2
    row, col = np.ogrid[:radius, :radius]
    dist_xj =  np.abs(row - center)+np.abs(col - center)
    for row in range(radius,im.shape[0]-radius):
        for col in range(radius,im.shape[1]-radius):
            window = im[row-int(radius/2):row+int(radius/2)+1,col-int(radius/2):col+int(radius/2)+1]
            gi = np.exp(-(np.square(window-im[row,col])/stdIntensity))
            gs = np.exp(-(np.square(dist_xj/stdSpatial)))
            sum_gs_gi = np.sum(gs*gi)
            final = np.sum(gs*gi*window)/sum_gs_gi
            bilateral_im[row,col] = np.uint8(np.sum((gs*gi/sum_gs_gi)*window))
    bilateral_im = np.uint8(cv2.bilateralFilter(im,radius,stdIntensity,stdSpatial))
    return bilateral_im


