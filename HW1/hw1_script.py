from hw1_functions import *

if __name__ == "__main__":

    path_image = r'Images\darkimage.tif'
    darkimg = cv2.imread(path_image)
    darkimg_gray = cv2.cvtColor(darkimg, cv2.COLOR_BGR2GRAY)

    print("Start running script  ------------------------------------\n")
    print_IDs()
    print("a ------------------------------------\n")
    enhanced_img, a, b = contrastEnhance(darkimg_gray, (0, 255))  # add parameters

    # display images
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(darkimg)
    plt.title('original')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_img, cmap='gray', vmin=0, vmax=255)
    plt.title('enhanced contrast')

    # print a,b
    print("a = {}, b = {}\n".format(a, b))

    # display mapping
    showMapping((np.min(darkimg_gray), np.max(darkimg_gray)), a, b)  # add parameters

    already_enhanced_img = cv2.imread(r"Images\lakeScene.tif")
    already_enhanced_img_gray = cv2.cvtColor(already_enhanced_img, cv2.COLOR_BGR2GRAY)

    print("b ------------------------------------\n")
    enhanced2_img, a, b = contrastEnhance(already_enhanced_img_gray, (0, 255))  # add parameters
    # print a,b
    print("enhancing an already enhanced image\n")
    print("a = {}, b = {}\n".format(a, b))

    # display the difference between the two image (Do not simply display both images)

    difference_img = enhanced2_img - already_enhanced_img_gray
    plt.imshow(difference_img)
    plt.show()

    print("c ------------------------------------\n")
    some_img = cv2.imread(r"Images\barbara.tif")
    some_img_gray = cv2.cvtColor(some_img, cv2.COLOR_BGR2GRAY)
    mdist = minkowski2Dist(some_img_gray, some_img_gray)  # add parameters
    print("Minkowski dist between image and itself\n")
    print("d = {}\n".format(mdist))

    # implement the loop that calculates minkowski distance as function of increasing contrast

    barbara_gray = some_img_gray
    step = (np.max(barbara_gray) - np.min(barbara_gray)) / 20
    X_axis = []
    Y_axis = []
    for k in range(1, 21):
        contrast = contrastEnhance(barbara_gray, ((np.min(barbara_gray)), np.min(barbara_gray) + k * step))[0]
        dists = minkowski2Dist(contrast, barbara_gray)
        X_axis.append(k * step)
        Y_axis.append(dists)
    plt.figure()
    plt.plot(X_axis, Y_axis)
    plt.xlabel("contrast")
    plt.ylabel("distance")
    plt.title("Minkowski distance as function of contrast")
    plt.show()

    print("d ------------------------------------\n")

    race_car_img = cv2.imread(r"Images\racecar.tif")
    race_car_gray = cv2.cvtColor(race_car_img, cv2.COLOR_BGR2GRAY)

    arr = np.zeros(shape=(256, 1))
    for i in range(256):
        arr[i, 0] = i

    slices_mat = sliceMat(race_car_gray)
    d = np.matmul(slices_mat, arr).reshape((256, 256)) - race_car_gray
    print("{}".format(d))
    print("Sum of all elements: {}\n".format(np.sum(d)))

    print("e ------------------------------------\n")

    im = cv2.imread(r"Images\lena.tif")
    im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    max_contrast_im, a, b = contrastEnhance(im_grey, (0, 255))
    slice_im = sliceMat(im_grey)
    tonemaped_im = np.matmul(slice_im, np.arange(256)).reshape(im_grey.shape)
    d = tonemaped_im - im_grey
    print("sum of diff between image and slices*[0..255] =\n {}".format(d))

    # then display
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_grey, cmap='gray', vmin=0, vmax=255)
    plt.title("original image")
    plt.subplot(1, 2, 2)
    plt.imshow(tonemaped_im, cmap='gray', vmin=0, vmax=255)
    plt.title("tone mapped")
    plt.show()

    print("f ------------------------------------\n")
    negative_im = sltNegative(darkimg_gray)
    plt.figure()
    plt.imshow(negative_im, cmap='gray', vmin=0, vmax=255)
    plt.title("negative image using SLT")
    plt.show()

    print("g ------------------------------------\n")
    thresh = 120  # change for different effect.
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    lena_gray.resize(lena.shape[0], lena.shape[1])
    thresh_im = sltThreshold(lena_gray, thresh)  # add parameters

    plt.figure()
    plt.imshow(thresh_im, cmap='gray', vmin=0, vmax=255)
    plt.title("thresh image using SLT")
    plt.show()

    print("h ------------------------------------\n")

    im1 = lena_gray
    im2 = cv2.imread(r"Images\stroller.tif")
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    SLTim = SLTmap(darkimg_gray, im2)[0]

    # then print
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(darkimg_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("Dark_image.tif")
    plt.subplot(1, 3, 2)
    plt.imshow(SLTim, cmap='gray', vmin=0, vmax=255)
    plt.title("Tone Mapped image")
    plt.subplot(1, 3, 3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=255)
    plt.title("Stroller.tif")

    print("MSD between STLmapped image and stroller.tif(im2) ={}\n".format(meanSqrDist(SLTim, im2)))
    print("MSD between lena.tif(im1) and stroller.tif(im2) ={}\n".format(meanSqrDist(im1, im2)))

    d1 = SLTmap(darkimg_gray, im2)[0]
    d2 = SLTmap(im2, darkimg_gray)[0]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(d1, cmap='gray', vmin=0, vmax=255)
    plt.title("im1 mapped into im2")
    plt.subplot(1, 2, 2)
    plt.imshow(d2, cmap='gray', vmin=0, vmax=255)
    plt.title("im2 mapped into im1")
    plt.show()

    print("i ------------------------------------\n")
    # prove computationally
    d = d2 - d1
    print(" {}".format(d))

    plt.show()
