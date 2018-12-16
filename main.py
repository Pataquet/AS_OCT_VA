import cv2
import numpy as np
import scipy.stats
from scipy.signal import argrelextrema

import positions

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

#interpolacion
# regresion robusta
#runsac
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
    dist = []

    for i in range(columnas):
        countR1 = 0
        posInit = 0
        blanco = True
        for j in range(filas):
            if(img[j][i] == 0):blanco = False
            if(blanco):continue

            if(img[j][i] == 255):
                break
            if(countR1 == 0):
                posInit = j
            totalPX = totalPX+1
            countR1=countR1+1
            out[j][i] = 255

        dist.append(positions.Positions(i, posInit, posInit + countR1))
        valueComunas[i]= countR1


    # print(dist[0].colum)
    plt.figure()
    plt.imshow(out, cmap='gray')
    plt.title('Parts'), plt.xticks([]), plt.yticks([])

    # print("Nº total de px: ",totalPX)
    # print("Media PX: ", totalPX/columnas)

    return out, dist

def calculateDist(dist):
    listInit = np.zeros(len(dist))
    listFin = np.zeros(len(dist))
    listDist = np.zeros(len(dist))

    for i in range(len(dist)):
        listInit[i]= dist[i].inicio
        listFin[i]= dist[i].fin
        listDist[i] = dist[i].distancia

    gaussInit = ndimage.gaussian_filter1d(listInit, 20)
    gaussFin = ndimage.gaussian_filter1d(listFin, 20)

    distancias = np.abs(np.subtract(gaussFin, gaussInit))
    for i in  range(len(distancias)):
        dist[i].distanciaSuav = distancias[i]

    plt.figure()
    plt.plot(distancias)
    plt.title("SUAVIZADAAAAAA")

    plt.figure()
    plt.plot(listDist)
    plt.title("SIIIIINNNNN")
    desviacionTipica(dist)

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
    size = len(dist)
    fivePercent =  int(size*0.05)
    fifteenPercent = int(size*0.15)
    twentyPercent = int(size*0.20)
    twentyFivePercent = int(size * 0.25)

    dist.sort(key=lambda x: x.distanciaSuav, reverse=True)
    dist = sorted(dist, key=lambda x: x.distanciaSuav, reverse=True)

    dTotal = np.var([f.distanciaSuav for f in dist])

    if (dTotal > 50):

        dPrin = np.var([f.distanciaSuav for f in dist[fivePercent:(size-fifteenPercent)]])
        dFin = np.var([f.distanciaSuav for f in dist[fifteenPercent:(size-fivePercent)]])
        if(dFin > dPrin):
            if(dPrin > 100):
                dPrin = np.var([f.distanciaSuav for f in dist[fivePercent:(size - twentyPercent)]])
                if(dPrin > 100):
                    dPrin = np.var([f.distanciaSuav for f in dist[fivePercent:(size - twentyFivePercent)]])
                    valuePrin = fivePercent
                    valueFin = twentyFivePercent
                else:
                    valuePrin = fivePercent
                    valueFin = twentyPercent
            else:
                valuePrin = fivePercent
                valueFin = fifteenPercent
        else :
            if (dFin > 100):
                dFin = np.var([f.distanciaSuav for f in dist[twentyPercent:(size - fivePercent)]])
                if (dFin> 100):
                    dFin = np.var([f.distanciaSuav for f in dist[twentyFivePercent:(size - fivePercent)]])
                    valuePrin = twentyFivePercent
                    valueFin = fivePercent
                else:
                    valuePrin = twentyPercent
                    valueFin = fivePercent
            else:
                valuePrin = fifteenPercent
                valueFin =fivePercent
    else :
        print("EEEEEEQQQQQQQQ")
        valuePrin = 0
        valueFin = 0
    print(valuePrin)
    print(valueFin)

    for i in range(valuePrin):
        dist[i].distanciaSuav = -1

    for i in range(size-valueFin, size):
        dist[i].distanciaSuav = -1


    dist.sort(key=lambda x: x.colum)
    dist = sorted(dist, key=lambda x: x.colum)

    tmp = np.zeros(int(size))

    for i in range(int(size)):
        tmp[i] = dist[i].distanciaSuav

    localMaxim =argrelextrema(tmp, np.greater)
    print(localMaxim[0])

    plt.figure()
    plt.plot([f.distanciaSuav for f in dist])
    plt.title("DESVIACION TIPICA")

    unionMaxLocal(localMaxim[0], dist)
    # print([f.distanciaSuav for f in dist])

def unionMaxLocal(localMax, points):
    sizePoints = int(len(points))
    sizeMax = int(len(localMax))
    localAntSig = []
    negativos = False
    for i in range(sizePoints):
        if (not negativos):
            if(points[i].distanciaSuav == -1):
                print(points[i].distanciaSuav)
                negativos= True
                if(i > localMax[sizeMax-1]):
                    localAntSig.append([localMax[sizeMax-1], localMax[sizeMax-1]])
                else:
                    for j in range(sizeMax):
                        if(localMax[j] > i):
                            if(j == 0):
                                localAntSig.append([localMax[j], localMax[j]])
                            else:
                                localAntSig.append([localMax[j-1], localMax[j]])
                            break
        else:
            if (points[i].distanciaSuav != -1):
                negativos = False

    print(localAntSig)

    for i in range(len(localAntSig)):
        points = lineFromPoints(localAntSig[i], localMax, points)

    plt.figure()
    plt.plot([f.distanciaSuav for f in points])
    plt.title("CREAR LINEAS")


def lineFromPoints(localAntSig, localMax, points):

    P = [localAntSig[0], points[localAntSig[0]].distanciaSuav]
    Q = [localAntSig[1], points[localAntSig[1]].distanciaSuav]

    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])

    if (localAntSig[0] != localAntSig[1]):
        print("DENTROOOOOOOOO")
        for i in range(localAntSig[0], localAntSig[1]):
            points[i].distanciaSuav =  (c - a*i)/b
    else:
        print("MAAAAAAAAAAAAAL")

        if(localMax[0] == localAntSig[1]):
            print("PRINCIPIOOOOO")
            for i in range(localAntSig[1]):
                points[i].distanciaSuav = points[localAntSig[0]].distanciaSuav
        elif(localMax[len(localMax)-1] == localAntSig[1]):
            print("FIIIIIIN")

            for i in range(localAntSig[1],len(points)):
                points[i].distanciaSuav = points[localAntSig[0]].distanciaSuav

    return points
    # if (b < 0):
    #     print("The line passing through points P and Q is:",
    #           a, "x ", b, "y = ", c, "\n")
    # else:
    #     print("The line passing through points P and Q is: ",
    #           a, "x + ", b, "y = ", c, "\n")



def reduceMiddleBorder(img, regions):
    # print(regions[1].coords)
    tmp = np.zeros(img.shape)
    a = newImage(tmp, regions[1].coords)
    # print(regions[1].coords)
    a = combine(img, a)
    filas, columnas = a.shape
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


def execute(th):
    # impTodas(th)
    imgOr, imgTh, imgDl = readImage(th, 'AS-OCT\im12.jpeg')
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

    calculateDist(capa2)
    # desviacionTipica(capa1)



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
