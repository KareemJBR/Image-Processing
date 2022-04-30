import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

import hw3_functions
from hw1_functions import contrastEnhance
from hw3_functions import cleanImageMedian


def print_IDs():
    print("205896103 , 211406343\n")


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def mapImage(im, T, sizeOutIm):
    im_new = im.min() * np.ones(sizeOutIm)

    # create meshgrid of all coordinates in new image [x,y]
    xx = np.arange(sizeOutIm[0])
    yy = np.arange(sizeOutIm[1])
    xx, yy = np.meshgrid(xx, yy)
    xx = xx.ravel()
    yy = yy.ravel()
    xy_coords = np.vstack((xx, yy))

    # add homogenous coord [x,y,1]
    N = sizeOutIm[0] * sizeOutIm[1]
    xy_coords = np.vstack((xy_coords, np.ones((1, N), int)))

    # calculate source coordinates that correspond to [x,y,1] in new image
    src_coords = np.matmul(np.linalg.inv(T), xy_coords)
    src_coords = np.transpose(src_coords)
    xy_coords = np.transpose(xy_coords)
    # find coordinates outside range and delete (in source and target)
    to_keep = src_coords.min(axis=1) >= 0
    src_coords = src_coords[to_keep, :]
    xy_coords = xy_coords[to_keep, :]
    to_keep = src_coords.max(axis=1) < im.shape[1]
    src_coords = src_coords[to_keep, :]
    xy_coords = xy_coords[to_keep, :]
    to_keep = src_coords.max(axis=1) < im.shape[0]
    src_coords = src_coords[to_keep, :]
    xy_coords = xy_coords[to_keep, :]

    # interpolate - bilinear
    xy_coords = xy_coords.astype(int)
    im_new[xy_coords[:, 0], xy_coords[:, 1]] = bilinear_interpolate(im, src_coords[:, 0], src_coords[:, 1])

    return np.transpose(im_new)


def minIMG(img):
    # Creates the shape of the kernel
    size = (3, 3)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    return cv2.erode(img, kernel)


def maxIMG(img):
    # Creates the shape of the kernel
    size = (3, 3)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # Applies the minimum filter with kernel NxN
    return cv2.dilate(img, kernel)


def getImagePts(im1, im2, varName1, varName2, nPoints):
    ones_ = np.ones((nPoints, 1)).astype(int)  # column of ones to append.

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.title('Select im1 Pts.')
    imagePts1 = np.round(plt.ginput(nPoints, 100)).astype(int)

    plt.close('all')

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title('Select im2 Pts.')
    imagePts2 = np.round(plt.ginput(nPoints, 100)).astype(int)

    plt.close('all')

    imagePts1 = np.hstack((imagePts1, ones_))
    imagePts2 = np.hstack((imagePts2, ones_))

    np.save(varName1 + ".npy", imagePts1)
    np.save(varName2 + ".npy", imagePts2)


def findAffineTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    X = np.zeros((N * 2, 6), dtype='float32')
    XD = np.zeros((N * 2, 1), dtype='float32')

    X[np.arange(2 * N) % 2 == 0, 4] = 1
    X[np.arange(2 * N) % 2 == 1, 5] = 1
    # iterate iver points to create x , x'
    for i in range(0, N):
        X[i * 2][0] = pointsSet1[i][0]  # x
        X[i * 2][1] = pointsSet1[i][1]  # y
        X[i * 2 + 1][2] = pointsSet1[i][0]  # x
        X[i * 2 + 1][3] = pointsSet1[i][1]  # y
        XD[i * 2, 0] = pointsSet2[i][0]
        XD[i * 2 + 1, 0] = pointsSet2[i][1]
    # calculate T - be careful of order when reshaping it
    pinv = np.linalg.pinv(X)
    T = np.matmul(pinv, XD)
    T[np.arange(6)] = T[[0, 1, 4, 2, 3, 5]]
    T = np.vstack((T.reshape((2, 3)), [0, 0, 1]))
    return T


def clean_im1(im):
    # getImagePts(im, im, "im1", "im2", 12)
    median_im = im.copy()
    radius = 3
    for row in range(radius, median_im.shape[0] - radius):
        for col in range(radius, median_im.shape[1] - radius):
            window = im[row - int(radius / 2):row + int(radius / 2) + 1,
                     col - int(radius / 2):col + int(radius / 2) + 1]
            median = np.uint8(np.median(window))
            median_im[row, col] = median

    T = findAffineTransform(np.load("im1.npy"), np.load("im2.npy"))
    median_im = mapImage(median_im, T, im.shape)
    return median_im


def clean_im2(im):
    clear_im = np.fft.fft2(im)
    clear_im[4, 28] = 0
    clear_im[252, 228] = 0
    return np.abs((np.fft.ifft2(clear_im)))


def clean_im3(im):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    clean_im = cv2.filter2D(im, -1, sharpen_kernel)
    return clean_im


def clean_im4(im):
    msk = np.zeros(im.shape)
    msk[1, 1] = 0.5
    msk[5, 80] = 0.5
    mskFFT = np.fft.fft2(msk)
    imFFT = np.fft.fft2(im)
    ind = (np.abs(mskFFT) < 0.01)
    mskFFT[ind] = 1
    cleared_im = imFFT / mskFFT
    return np.abs(np.fft.ifft2(cleared_im))


def clean_im5(im):
    return cleanImageMedian(maxIMG(minIMG(minIMG(maxIMG(im)))), 3)


def clean_im6(im):
    FFT = np.fft.fftshift(np.fft.fft2(im))
    FFT[108:149, 108:149] = 2 * FFT[108:149, 108:149]
    clean_im = np.abs(np.fft.ifft2(np.fft.ifftshift(FFT)))
    clean_im = hw3_functions.cleanImageMean(clean_im, 4, 5)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    clean_im = cv2.filter2D(clean_im, -1, sharpen_kernel)

    return clean_im


def clean_im7(im):
    msk = np.zeros(im.shape)
    msk[0, 0:10] = 0.1
    mskFFT = np.fft.fft2(msk)
    ind = (np.abs(mskFFT) <= 0.01)
    mskFFT[ind] = 1
    return np.abs(np.fft.ifft2(np.fft.fft2(im) / mskFFT))


def clean_im8(im):
    return contrastEnhance(im, (0, 255))[0]


'''
    # an example of how to use fourier transform:
    img = cv2.imread(r'Images/windmill.tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_fourier = np.fft.fftshift(np.fft.fft2(img))

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('original image')

    plt.subplot(1,3,2)
    plt.imshow(np.log(abs(img_fourier)), cmap='gray')
    plt.title('fourier transform of image')

    img_inv = np.fft.ifft2(img_fourier)
    plt.subplot(1,3,3)
    plt.imshow(abs(img_inv), cmap='gray')
    plt.title('inverse fourier of the fourier transform')

'''
