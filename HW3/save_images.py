import cv2.cv2 as cv2
from hw3_functions import *


def save_Imgs():
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 1 ----------------------------------------------------------
    lena_sp_low = addSPnoise(lena_gray, 0.01)
    np.save("SPlow", lena_sp_low)
    np.save("lena_median_SPLow.tif", cleanImageMedian(lena_sp_low, 3))
    np.save("lena_mean_SPLow.tif", cleanImageMean(lena_sp_low, 5, 5.))
    np.save("lena_blf_SPLow.tif", bilateralFilt(lena_sp_low, 5, 10., 10.))

    # 2 ----------------------------------------------------------
    lena_sp_high = addSPnoise(lena_gray, 0.2)
    np.save("SPhigh", lena_sp_high)
    np.save("lena_median_SPHigh.tif", cleanImageMedian(lena_sp_high, 5))
    np.save("lena_mean_SPHigh.tif", cleanImageMean(lena_sp_high, 5, 5.))
    np.save("lena_blf_SPHigh.tif", bilateralFilt(lena_sp_high, 9, 20., 20.))

    # 3 ----------------------------------------------------------
    lena_gaussian = addGaussianNoise(lena_gray, 10.)
    np.save("gaussianLow", lena_gaussian)
    np.save("lena_median_gaussianLow.tif", cleanImageMedian(lena_gaussian, 5))
    np.save("lena_mean_gaussianLow.tif", cleanImageMean(lena_gaussian, 5, 3.))
    np.save("lena_blf_gaussianLow.tif", bilateralFilt(lena_gaussian, 9, 20., 20.))

    # 4 ----------------------------------------------------------

    lena_gaussian = addGaussianNoise(lena_gray, 25.)
    np.save("gaussianHigh", lena_gaussian)
    np.save("lena_median_gaussianHigh.tif", cleanImageMedian(lena_gaussian, 7))
    np.save("lena_mean_gaussianHigh.tif", cleanImageMean(lena_gaussian, 7, 5.))
    np.save("lena_blf_gaussianHigh.tif", bilateralFilt(lena_gaussian, 9, 40., 40.))
