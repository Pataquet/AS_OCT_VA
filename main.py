import cv2
import numpy as np
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


def readImage (thigh, img):
    imgOr = cv2.imread(img, 0)

    imgTh = cv2.imread(img, 0)
    imgTh = ndimage.grey_opening(imgTh, size=(3, 4))

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

    kernel = np.ones((1, 5))
    imgTh = cv2.erode(imgTh, kernel, iterations=1)

    kernel = np.ones((2,15))
    imgDl = cv2.dilate(imgTh, kernel, iterations = 1)



    return imgOr, imgTh, imgDl

def newImage(image, regionCoords):
    for i in regionCoords:
        image[i[0]][i[1]] = 1
    return image

def cConexas(image):

    label_image = label(image)
    imageComponentesConexas = np.zeros(image.shape)

    regions = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 1000:
            regions.append(region)

    for region in regions:
        imageComponentesConexas = newImage(imageComponentesConexas, region.coords)




    kernel = np.ones((3, 9))
    imageComponentesConexas = cv2.morphologyEx(imageComponentesConexas, cv2.MORPH_CLOSE, kernel)



    label_image = label(imageComponentesConexas)
    imageComponentesConexas = np.zeros(image.shape)

    regions = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 2000:
            regions.append(region)

    regions.sort(key=lambda x: x.area, reverse=True)
    regions = sorted(regions, key=lambda x: x.area, reverse=True)

    regions = regions[0:5]

    for region in regions:
        imageComponentesConexas = newImage(imageComponentesConexas, region.coords)

    return imageComponentesConexas, regions

# def unionCC(regions):
#
#     for r in regions:
#         minr1, minc1, maxr1, maxc1 = regions[r].bbox
#         ltmp = []
#         for i in range(len(regions)):
#             minr2, minc2, maxr2, maxc2 = regions[i].bbox
#             if(minc1==minc2 and minr1 == minr2 and maxc1== maxc2 and maxr1== maxr2 ): continue
#             if (minr1<minr2 and maxr1> maxr2):
#                 ltmp.append()

        


def cConexas2(image, nombre):

    label_image = label(image)
    image_label_overlay = label2rgb(image, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))

    imageComponentesConexas = np.zeros(image.shape)

    ax.imshow(image_label_overlay)
    i = 0
    regions = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        regions.append(region)
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        if (i == 0):
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        elif (i == 1):
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='green',
                                      linewidth=2)
        elif (i == 2):
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='blue',
                                      linewidth=2)
        elif (i == 3):
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='yellow',
                                      linewidth=2)
        else:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='purple',
                                      linewidth=2)

        i = i + 1
        ax.add_patch(rect)

        imageComponentesConexas = newImage(imageComponentesConexas, region.coords)


    ax.set_axis_off()
    plt.tight_layout()
    plt.imshow(imageComponentesConexas, cmap='gray')
    plt.title(nombre)
    return imageComponentesConexas, regions

def showImage(ori, th,  dl, cc):
    plt.figure()
    plt.subplot(221), plt.imshow(ori, cmap='gray')
    plt.title('OR Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(th, cmap='gray')
    plt.title('th Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(dl, cmap='gray')
    plt.title('DL Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(cc, cmap='gray')
    plt.title('CC Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def combine(original, regions):
    filas, columnas = original.shape

    # outImage = np.zeros((filas,columnas))
    outImage = original.copy()
    for i in range(filas):
        for j in range(columnas):
            if (regions[i][j] == 1 ):
                outImage[i][j] = original[i][j]
            else:
                outImage[i][j] =0

    # kernel = np.ones((3,4))
    # print(kernel)
    # outImage =  cv2.morphologyEx(outImage, cv2.MORPH_CLOSE, kernel)
    # outImage = cv2.Canny(outImage, 120, 180)


    return outImage

def countPx(img):
    filas, columnas = img.shape
    out = img.copy()
    totalPX = 0
    valueComunas = np.zeros(columnas)
    for i in range(columnas):
        countR1 = 0

        blanco = True
        for j in range(filas):
            if(img[j][i] == 0):blanco = False
            if(blanco):continue

            if(img[j][i] == 255):
                break
            totalPX = totalPX+1
            countR1=countR1+1
            out[j][i] = 255
        valueComunas[i]= countR1

    plt.figure()
    plt.imshow(out, cmap='gray')
    plt.title('Parts'), plt.xticks([]), plt.yticks([])

    print("Nº total de px: ",totalPX)
    print("Media PX: ", totalPX/columnas)
    return out, valueComunas

def reduceBorders(img):
    filas, columnas = img.shape
    out = np.zeros((filas, columnas))

    # plt.figure()
    # plt.imshow(img, cmap='gray')
    # plt.title('IMG'), plt.xticks([]), plt.yticks([])

    for i in range(columnas):
        blanco = True
        for j in range(filas):
            if(img[j][i] == 0): blanco = False
            if(blanco): continue
            if((not blanco) and img[j][i] == 255):
                out[j][i] = 255
                blanco= True
    plt.figure()
    plt.imshow(out, cmap='gray')
    plt.title('Reduc'), plt.xticks([]), plt.yticks([])

    return out

def desviacionTipica(dist):


    dist = np.sort(dist)
    size = len(dist)
    fivePercent =  int(size*0.05)
    fifteenPercent = int(size*0.15)


    print(dist[fivePercent:(size-fifteenPercent)])
    print(dist[fifteenPercent:(size-fivePercent)])
    # print(fivePercent)
    # print(fifteenPercent)
    # print(size)

    dTotal = np.var(dist)
    dPrin = np.var(dist[fivePercent:(size-fifteenPercent)])
    dFin = np.var(dist[fifteenPercent:(size-fivePercent)])
    dEq = np.var(dist[fivePercent:(size-fivePercent)])

    print(dTotal)
    print(dPrin)
    print(dFin)
    print(dEq)


def reduceMiddleBorder(img, regions):
    # print(regions[1].coords)
    tmp = np.zeros(img.shape)
    a = newImage(tmp, regions[1].coords)
    # print(regions[1].coords)
    a = combine(img, a)
    filas, columnas =  a.shape
    for i in range(columnas):
        max = 0
        f = 0
        c = 0
        for j in range(filas):
            if (a[j][i]> max):
                max = a[j][i]
                f = j
                c = i
            a[j][i] =0
        if(f != 0):
            # img[f][c] = 255
            tmp[f][c] = 255

    for i in regions[1].coords:
        img[i[0]][i[1]] = 0

    for i in range(filas):
        for j in range(columnas):
            if(tmp[i][j] == 255):
                img[i][j]=255
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('MIDDLE'), plt.xticks([]), plt.yticks([])

    return img

def a(img):

    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
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
    plt.title('Drawing'), plt.xticks([]), plt.yticks([])



def execute(th):
    # impTodas(th)
    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im3.jpeg')
    imgCc, _ = cConexas(imgDl)


    imgCc, regiones = cConexas2(imgCc, "IMG 4")



    # imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im9.jpeg')
    # imgCc, _ = cConexas(imgDl)
    # imgCc, regiones = cConexas2(imgCc, "IMG 9")
    #
    # imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im12.jpeg')
    # imgCc, _ = cConexas(imgDl)
    # imgCc, regiones = cConexas2(imgCc, "IMG 12")


    # plt.figure()
    # plt.imshow(imgDl, cmap='gray')
    # plt.title('D1'), plt.xticks([]), plt.yticks([])

    imgCombine = combine(imgOr, imgCc)
    # a(imgCombine)

    rmb = reduceMiddleBorder(imgCombine,regiones)

    rmb[rmb > 120] = 255
    rmb[rmb  <= 120] = 0
    imgReduc = reduceBorders(rmb )

    imgFin, _ = countPx(imgReduc)
    imgFin, capa1 = countPx(imgFin)
    imgFin, capa2 = countPx(imgFin)

    desviacionTipica(capa1)



    # mediaTruncada(capa1)
    # mediaTruncada(capa2)


    # imagem = cv2.bitwise_not(imgCombine)
    # imgFin = cConexas(imagem, True)

    # plt.figure()
    # plt.plot(imgOr[:, 0])

    # imgFin = countPx(imgCombine)
    # imgFin = countPx(imgFin)
    # imgFin = countPx(imgFin)
    # imgFin = countPx(imgFin)

    # edges = cv2.Canny(imgCombine, 0, 50)
    # kernel = np.ones((2, 1))

    # erode = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    # tmp = countPx(edges)




    # plt.figure()
    # plt.imshow(imgCombine, cmap='gray')
    # plt.title('comb2'), plt.xticks([]), plt.yticks([])



    # showImage(imgOr, imgTh, imgDl, imgCc)

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




def impTodas(th):

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im1.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 1")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im2.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 2")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im3.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 3")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im4.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 4")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im5.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 5")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im6.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 6")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im7.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 7")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im8.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 8")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im9.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 9")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im10.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 10")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im11.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 11")

    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im12.jpeg')
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc, "IMG 12")

execute(80)
plt.show()
