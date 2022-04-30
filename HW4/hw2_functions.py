import numpy as np
import cv2
import matplotlib.pyplot as plt


def writeMorphingVideo(image_list, video_name):
    out = cv2.VideoWriter(video_name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, image_list[0].shape, 0)
    for im in image_list:
        out.write(im)
    out.release()


def createMorphSequence(im1, im1_pts, im2, im2_pts, t_list, transformType):
    T12 = np.eye(3)
    T21 = np.eye(3)
    I = np.eye(3)
    if transformType:
        T12 = findProjectiveTransform(im1_pts, im2_pts)
        T21 = findProjectiveTransform(im2_pts, im1_pts)
    else:
        T12 = findAffineTransform(im1_pts, im2_pts)
        T21 = findAffineTransform(im2_pts, im1_pts)
    ims = []
    T12_t = np.eye(3)
    T21_t = np.eye(3)
    for t in t_list:
        T12_t = (1 - t) * I + t * T12
        T21_t = (1 - t) * T21 + t * I
        im1_new = mapImage(im1, T12_t, im2.shape)
        im2_new = mapImage(im2, T21_t, im1.shape)
        nim = (1 - t) * im1_new + t * im2_new
        # plt.plot(1,1,1)
        # plt.imshow(nim, cmap='gray', vmin=0, vmax=255)
        # plt.show()
        ims.append(nim.astype(np.uint8))
    return ims


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

    # apply corresponding coordinates
    # new_im [ target coordinates ] = old_im [ source coordinates ]
    return np.transpose(im_new)


def findProjectiveTransform(pointsSet1, pointsSet2):
    N = pointsSet1.shape[0]
    X = np.zeros((N * 2, 8), dtype='float32')
    XD = np.zeros((N * 2, 1), dtype='float32')

    # iterate iver points to create x , x'
    X[np.arange(2 * N) % 2 == 0, 4] = 1
    X[np.arange(2 * N) % 2 == 1, 5] = 1
    for i in range(0, N):
        X[i * 2][0] = pointsSet1[i][0]  # x
        X[i * 2][1] = pointsSet1[i][1]  # y

        X[i * 2][6] = -1 * pointsSet1[i][0] * pointsSet2[i, 0]  # x*x'
        X[i * 2][7] = -1 * pointsSet1[i][1] * pointsSet2[i, 0]  # y*x'

        X[i * 2 + 1][2] = pointsSet1[i][0]  # x
        X[i * 2 + 1][3] = pointsSet1[i][1]  # y

        X[i * 2 + 1][6] = -1 * pointsSet1[i][0] * pointsSet2[i, 1]  # x*y'
        X[i * 2 + 1][7] = -1 * pointsSet1[i][1] * pointsSet2[i, 1]  # y*y'

        XD[i * 2, 0] = pointsSet2[i][0]
        XD[i * 2 + 1, 0] = pointsSet2[i][1]
    # calculate T - be careful of order when reshaping it
    pinv = np.linalg.pinv(X)
    T = np.matmul(pinv, XD)
    T[np.arange(8)] = T[[0, 1, 4, 2, 3, 5, 6, 7]]
    T = np.append(T, [1]).reshape((3, 3))
    return T


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


def getImagePts(im1, im2, varName1, varName2, nPoints):
    ones_ = np.ones((nPoints, 1)).astype(int)  # column of ones to append.

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=255)
    plt.title('Select im1 Pts.')
    imagePts1 = np.round(plt.ginput(nPoints, 100)).astype(int)
    plt.show()

    imagePts1 = np.hstack((imagePts1, ones_))
    np.save(varName1 + ".npy", imagePts1)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title('Select im2 Pts.')
    imagePts2 = np.round(plt.ginput(nPoints, 100)).astype(int)
    plt.show()

    imagePts2 = np.hstack((imagePts2, ones_))
    np.save(varName2 + ".npy", imagePts2)
