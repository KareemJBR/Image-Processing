from save_images import *

if __name__ == "__main__":
    # feel free to load different image than lena
    lena = cv2.imread(r"Images\lena.tif")
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    # save_Imgs()
    # 1 ----------------------------------------------------------
    # add salt and pepper noise - low

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(np.load("SPlow.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - low")
    plt.subplot(2, 3, 4)
    plt.imshow(np.load("lena_median_SPLow.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(np.load("lena_mean_SPLow.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(np.load("lena_blf_SPLow.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")
    print("Conclusions -----  \n")
    print("Denoising Ranking:")
    print("1.Median denoising its better than Gaussian's because of the weight given to its neighbouring pixels"
          "\n2.Gaussian denosing"
          "\n3.BLT filter (BLT always gonna rank last because of it's \"saving\" the language feature.")
    plt.show()

    # 2 ----------------------------------------------------------
    # add salt and pepper noise - high

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(np.load("SPhigh.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("salt and pepper - low")
    plt.subplot(2, 3, 4)
    plt.imshow(np.load("lena_median_SPHigh.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(np.load("lena_mean_SPHigh.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(np.load("lena_blf_SPHigh.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")
    print("Conclusions -----  TODO: add explanation \n")
    print("Denoising Ranking:")
    print("1.Median denoising its better than Gaussian's because of the weight given to its neighbouring pixels"
          "\n2.Gaussian denosing"
          "\n3.BLT filter (BLT always gonna rank last because of it's \"saving\" the language feature.")
    plt.show()

    # 3 ----------------------------------------------------------
    # add gaussian noise - low

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(np.load("gaussianLow.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - low")
    plt.subplot(2, 3, 4)
    plt.imshow(np.load("lena_median_gaussianLow.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(np.load("lena_mean_gaussianLow.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(np.load("lena_blf_gaussianLow.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")
    print("Conclusions -----  TODO: add explanation \n")
    print("Denoising Ranking:")
    print("1.BLT filter"
          "\n2.Gaussian denosing"
          "\n3.Median denoising")
    plt.show()

    # 4 ----------------------------------------------------------
    # add gaussian noise - high

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title("original")
    plt.subplot(2, 3, 2)
    plt.imshow(np.load("gaussianHigh.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("gaussian noise - high")
    plt.subplot(2, 3, 4)
    plt.imshow(np.load("lena_median_gaussianHigh.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("median")
    plt.subplot(2, 3, 5)
    plt.imshow(np.load("lena_mean_gaussianHigh.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("mean")
    plt.subplot(2, 3, 6)
    plt.imshow(np.load("lena_blf_gaussianHigh.tif.npy"), cmap='gray', vmin=0, vmax=255)
    plt.title("bilateral")

    print("Conclusions -----  TODO: add explanation \n")
    print("Denoising Ranking:")
    print("1.BLT filter"
          "\n2.Gaussian denosing"
          "\n3.Median denoising")
    plt.show()
