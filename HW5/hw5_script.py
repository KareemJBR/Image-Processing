
from hw5_functions import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("----------------------------------------------------\n")
    print_IDs()
    print("----------------------------------------------------\n")

    im = cv2.imread(r'balls1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)

    plt.subplot(1, 2, 2)
    plt.imshow(sobel_edging(im), cmap='gray', vmin=0, vmax=255)

    plt.show()

    im = cv2.imread(r'coins1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.Canny(im, 100, 150, 7), cmap='gray', vmin=0, vmax=255)

    im = cv2.imread(r'balls1.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cv2.Canny(im, 100, 190, 5), cmap='gray', vmin=0, vmax=255)

    im = cv2.imread(r'balls2.tif')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.Canny(im, 55, 150, 5), cmap='gray', vmin=0, vmax=255)

    im = cv2.imread(r'boxOfChocolates1.tif')
    im2 = cv2.imread(r'boxOfChocolates2.tif')
    im3 = cv2.imread(r'boxOfChocolates2rot.tif')

    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

    im1_copy = im.copy()
    im2_copy = im2.copy()
    im3_copy = im3.copy()

    # low_val, high_val = 150, 250

    ShowLines(im1_copy, 150, 350)
    ShowLines(im2_copy, 100, 250)
    ShowLines(im3_copy, 100, 200)

    plt.show()
