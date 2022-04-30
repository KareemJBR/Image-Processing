import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np

from hw2_functions import *

if __name__ == "__main__":

    print("Students IDs:")
    print("205896103 , 211406343\n")

    im1 = cv2.imread(r'FaceImages\Face5.tif')
    im2 = cv2.imread(r'FaceImages\Face6.tif')
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # getImagePts(im1,im2,"a","b",12)
    T = findAffineTransform(np.load("a.npy"), np.load("b.npy"))
    # T = findProjectiveTransform(np.load("a.npy"),np.load("b.npy"))
    X = createMorphSequence(im1, np.load("a.npy"), im2, np.load("b.npy"), np.linspace(0, 1, 100), 0)
    writeMorphingVideo(X, "morph_video")

    lightHouse = cv2.imread('lighthouse.tif')
    mountainScene = cv2.imread('mountainScene.tif')

    lightHouse = cv2.cvtColor(lightHouse, cv2.COLOR_BGR2GRAY)
    mountainScene = cv2.cvtColor(mountainScene, cv2.COLOR_BGR2GRAY)

    # getImagePts(lightHouse, mountainScene, "lightHouse_points", "mountainScene_points", 12)

    proj_test = findProjectiveTransform(np.load("lightHouse_points.npy"), np.load("mountainScene_points.npy"))
    affine_test = findAffineTransform(np.load("lightHouse_points.npy"), np.load("mountainScene_points.npy"))

    coins_proj = mapImage(lightHouse, proj_test, lightHouse.shape)
    coins_affine = mapImage(lightHouse, affine_test, lightHouse.shape)

    titles = ['Projective', 'Affine']
    images = [coins_proj, coins_affine]

    fig_a, axes_a = plt.subplots(nrows=1, ncols=2)

    for i, ax in enumerate(axes_a.flat, start=1):
        ax.set_title(titles[i - 1])
        ax.imshow(images[i - 1], cmap='gray', vmin=0, vmax=255)

    plt.show()

    face1 = cv2.imread(r'FaceImages\Face5.tif')
    face2 = cv2.imread(r'FaceImages\Face6.tif')
    face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    # getImagePts(face1, face2, "face1", "face2", 4)

    T1 = findProjectiveTransform(np.load("a.npy"), np.load("b.npy"))
    T2 = findAffineTransform(np.load("a.npy"), np.load("b.npy"))
    proj = mapImage(face1, T1, im1.shape)
    affine = mapImage(face1, T2, im1.shape)
    print("Projective vs Affine Transformation")
    title = ["Face1", "Face2", "Projective_t", "Affine_t"]
    imgs = [face1, face2, proj, affine]
    fig, axes = plt.subplots(nrows=2, ncols=2)

    for i, ax in enumerate(axes.flat, start=1):
        ax.set_title(title[i - 1])
        ax.imshow(imgs[i - 1], cmap='gray', vmin=0, vmax=255)

    plt.show()

    face3 = cv2.imread(r'FaceImages\Face3.tif')
    face4 = cv2.imread(r'FaceImages\Face4.tif')
    face3 = cv2.cvtColor(face3, cv2.COLOR_BGR2GRAY)
    face4 = cv2.cvtColor(face4, cv2.COLOR_BGR2GRAY)

    # getImagePts(face3, face4, "face3_4", "face4_4", 4)
    # getImagePts(face3, face4, "face3_12", "face4_12", 12)
    # getImagePts(face3, face4, "face3_12_loc", "face4_12_loc", 12)
    print("First row shows pictures with 4 points choosen (eyes, nose and the mouth).")
    print("Second row shows pictures with 12 points choosen according to Location.jpeg")
    print("Third row shows pictures with 12 points choosen according to Location2.jpeg")
    print("added the cross disolve between pictures for easier comparison.")

    T12_4 = 0.5 * np.eye(3) + 0.5 * findAffineTransform(np.load("face3_4.npy"), np.load("face4_4.npy"))
    T21_4 = 0.5 * np.eye(3) + 0.5 * findAffineTransform(np.load("face4_4.npy"), np.load("face3_4.npy"))
    T12_12 = 0.5 * np.eye(3) + 0.5 * findAffineTransform(np.load("face3_12.npy"), np.load("face4_12.npy"))
    T21_12 = 0.5 * np.eye(3) + 0.5 * findAffineTransform(np.load("face4_12.npy"), np.load("face3_12.npy"))
    T12_12_loc = 0.5 * np.eye(3) + 0.5 * findAffineTransform(np.load("face3_12_loc.npy"), np.load("face4_12_loc.npy"))
    T21_12_loc = 0.5 * np.eye(3) + 0.5 * findAffineTransform(np.load("face4_12_loc.npy"), np.load("face3_12_loc.npy"))

    im1_4 = mapImage(face3, T12_4, face3.shape)
    im2_4 = mapImage(face4, T21_4, face4.shape)
    img_4 = 0.5 * im1_4 + 0.5 * im2_4
    im1_12 = mapImage(face3, T12_12, face3.shape)
    im2_12 = mapImage(face4, T21_12, face4.shape)
    img_12 = 0.5 * im1_12 + 0.5 * im2_12
    im1_12_loc = mapImage(face3, T12_12_loc, face3.shape)
    im2_12_loc = mapImage(face4, T21_12_loc, face4.shape)
    img_12_loc = 0.5 * im1_12_loc + 0.5 * im2_12_loc

    imgs = [im1_4, img_4, im2_4,
            im1_12, img_12, im2_12,
            im1_12_loc, img_12_loc, im2_12_loc]
    title = ["4 points", "4 points Cross", "4 points",
             "12 points", "12 points Cross", "12 points",
             "12 points loc.", "12 points loc. Cross", "12 points loc."]

    fig, axes = plt.subplots(nrows=3, ncols=3)

    for i, ax in enumerate(axes.flat, start=1):
        ax.set_title(title[i - 1])
        ax.imshow(imgs[i - 1], cmap='gray', vmin=0, vmax=255)

    plt.show()
