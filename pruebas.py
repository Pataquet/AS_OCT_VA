import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


def readImage (thigh, img):
    imgOr = cv2.imread(img, 0)

    median = cv2.medianBlur(imgOr,11)

    diff  = np.abs(np.subtract(median, imgOr))


    blur = cv2.blur(imgOr, (3, 3))  # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    # Finding contours for the thresholded image
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0)  # green - color for contours
        color = (255, 0, 0)  # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)

    plt.figure()
    plt.imshow(drawing, cmap='gray')
    plt.title('MEDIAN'), plt.xticks([]), plt.yticks([])
    #
    # plt.figure()
    # plt.imshow(diff, cmap='gray')
    # plt.title('DIFF'), plt.xticks([]), plt.yticks([])
    #
    # plt.figure()
    # plt.imshow(imgOr, cmap='gray')
    # plt.title('ORIG'), plt.xticks([]), plt.yticks([])

    imgTh = cv2.imread(img, 0)
    imgTh = cv2.GaussianBlur(imgTh, (11, 11), 0)
    #imgTh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # imgTh =  cv2.getGaussianKernel()
    # grad_y = cv2.Sobel(imgTh, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # grad_x = cv2.Sobel(imgTh, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    #
    # imgTh = cv2.convertScaleAbs(grad_y)
    # abs_grad_x = cv2.convertScaleAbs(grad_x)
    # abs_grad_y = cv2.convertScaleAbs(grad_y)
    # imgTh = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    imgTh[imgTh < thigh] = 0
    imgTh[imgTh >= thigh] = 1

    kernel = np.ones((1,14))

    imgDl = cv2.dilate(imgTh, kernel, iterations = 1)

    return imgOr, imgTh, imgDl

def newImage(image, regionCoords):
    for i in regionCoords:
        image[i[0]][i[1]] = 1
    return image

def cConexas(image):

    label_image = label(image)
    image_label_overlay = label2rgb(image, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))

    imageComponentesConexas = np.zeros(image.shape)

    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 1000:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='green', linewidth=2)
            print(minc)
            print(minr)
            print(maxc)
            print(maxr)

            # if(minc<50):
            ax.add_patch(rect)
            imageComponentesConexas = newImage(imageComponentesConexas, region.coords)


    ax.set_axis_off()
    plt.tight_layout()
    plt.imshow(imageComponentesConexas, cmap='gray')

    return imageComponentesConexas


def showImage(ori, th, cc, dl):
    plt.figure()
    plt.subplot(221), plt.imshow(ori, cmap='gray')
    plt.title('OR Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(th, cmap='gray')
    plt.title('th Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(dl, cmap='gray')
    plt.title('DL Image'), plt.xticks([]), plt.yticks([])
    plt.show()

    plt.subplot(224), plt.imshow(cc, cmap='gray')
    plt.title('CC Image'), plt.xticks([]), plt.yticks([])


def execute(th):
    imgOr, imgTh, imgDl  = readImage(th, 'AS-OCT\im3.jpeg')
    # imgCc = cConexas(imgDl)
    # showImage(imgOr, imgTh, imgCc, imgDl)

    # imgOr, imgTh = readImage(th, 'AS-OCT\im2.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im3.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im4.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im5.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im6.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im7.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im8.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im9.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im10.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im11.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)
    #
    # imgOr, imgTh = readImage(th, 'AS-OCT\im12.jpeg')
    # imgCc = cConexas(imgTh)
    # showImage(imgOr, imgTh, imgCc)


execute(90)
plt.show()
