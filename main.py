import cv2
import numpy as np
from scipy.signal import argrelextrema

import positions
import mostrarPasos as m

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

from skimage.measure import label, regionprops
from skimage.color import label2rgb


def readImage (thigh, img):
    imgOr = cv2.imread(img, 0)

    imgTh = cv2.imread(img, 0)

    opening = ndimage.grey_opening(imgTh, size=(3, 4))

    gauss = cv2.GaussianBlur(opening, (11, 11), 0)

    imgTh[gauss < thigh] = 0
    imgTh[gauss >= thigh] = 1

    kernel = np.ones((1, 5))
    erode = cv2.erode(imgTh, kernel, iterations=1)

    kernel = np.ones((2,15))
    dilate = cv2.dilate(erode, kernel, iterations = 1)

    m.showImage6(imgOr, opening, gauss, imgTh, erode, dilate, "Original", "Apertura gris", "Suav Gauss", "Th", "Erode", "Dilate")

    return imgOr, dilate

def newImage(image, regionCoords):
    for i in regionCoords:
        image[i[0]][i[1]] = 1
    return image

def cConexas(image):

    label_image = label(image)
    imageComponentesConexas = np.zeros(image.shape)

    regions1 = []
    for region in regionprops(label_image):
        if region.area >= 1000:
            regions1.append(region)

    for region in regions1:
        imageComponentesConexas = newImage(imageComponentesConexas, region.coords)

    kernel = np.ones((3, 9))
    imageComponentesConexasOpen = cv2.morphologyEx(imageComponentesConexas, cv2.MORPH_CLOSE, kernel)

    label_image = label(imageComponentesConexasOpen)
    imageComponentesConexas2 = np.zeros(image.shape)

    regions2 = []
    for region in regionprops(label_image):
        if region.area >= 2000:
            regions2.append(region)

    regions2.sort(key=lambda x: x.area, reverse=True)
    regions = sorted(regions2, key=lambda x: x.area, reverse=True)

    regions = regions[0:5]

    for region in regions:
        imageComponentesConexas2 = newImage(imageComponentesConexas2, region.coords)

    m.showImage3(imageComponentesConexas, imageComponentesConexasOpen, imageComponentesConexas2, "CC Area 1000", "CC Close", "CC Area 2000")

    return imageComponentesConexas2, regions

def cConexas2(image):

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
    plt.title("CC Finales")
    return imageComponentesConexas, regions

def combine(original, regions):
    filas, columnas = original.shape

    outImage = original.copy()
    for i in range(filas):
        for j in range(columnas):
            if (regions[i][j] == 1 ):
                outImage[i][j] = original[i][j]
            else:
                outImage[i][j] =0

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

    dt , ml = desviacionTipica(dist)
    return listDist, distancias, dt, ml

def reduceBorders(img):
    filas, columnas = img.shape
    out = np.zeros((filas, columnas))

    for i in range(columnas):
        blanco = True
        for j in range(filas):
            if(img[j][i] == 0): blanco = False
            if(blanco): continue
            if((not blanco) and img[j][i] == 255):
                out[j][i] = 255
                blanco= True
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


    desviacionTipica= [f.distanciaSuav for f in dist]
    union = unionMaxLocal(localMaxim[0], dist)

    return desviacionTipica, union

def unionMaxLocal(localMax, points):
    sizePoints = int(len(points))
    sizeMax = int(len(localMax))
    localAntSig = []
    negativos = False
    for i in range(sizePoints):
        if (not negativos):
            if(points[i].distanciaSuav == -1):
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

    union = [f.distanciaSuav for f in points]
    return union


def lineFromPoints(localAntSig, localMax, points):

    P = [localAntSig[0], points[localAntSig[0]].distanciaSuav]
    Q = [localAntSig[1], points[localAntSig[1]].distanciaSuav]

    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b * (P[1])

    if (localAntSig[0] != localAntSig[1]):
        for i in range(localAntSig[0], localAntSig[1]):
            points[i].distanciaSuav =  (c - a*i)/b
    else:
        if(localMax[0] == localAntSig[1]):
            for i in range(localAntSig[1]):
                points[i].distanciaSuav = points[localAntSig[0]].distanciaSuav
        elif(localMax[len(localMax)-1] == localAntSig[1]):
            for i in range(localAntSig[1],len(points)):
                points[i].distanciaSuav = points[localAntSig[0]].distanciaSuav

    return points

def reduceMiddleBorder(img, regions):
    tmp = np.zeros(img.shape)
    a = newImage(tmp, regions[1].coords)
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
            tmp[f][c] = 255

    for i in regions[1].coords:
        img[i[0]][i[1]] = 0

    for i in range(filas):
        for j in range(columnas):
            if(tmp[i][j] == 255):
                img[i][j]=255
    return img


def execute(imagen):
    imgOr, imgDl = readImage(80, imagen)
    imgCc, _ = cConexas(imgDl)
    imgCc, regiones = cConexas2(imgCc)

    imgCombine = combine(imgOr, imgCc)
    rmb = reduceMiddleBorder(imgCombine,regiones)
    thHigh = rmb.copy()
    thHigh[rmb > 120] = 255
    thHigh[rmb  <= 120] = 0
    imgReduc = reduceBorders(thHigh )

    m.showImage4(imgCombine, rmb, thHigh, imgReduc, "OR + Reg", "R mid edge", "ThHigh", "R All Edges")

    imgReg1, _ = countPx(imgReduc)
    imgReg2, capa1 = countPx(imgReg1)
    imgReg3, capa2 = countPx(imgReg2)

    m.showImage3(imgReg1, imgReg2, imgReg3, "PX R1", "PX R2", "PX R3")

    distOr, distSuv, distDP, distML = calculateDist(capa2)

    m.showStadistics(distOr, distSuv, distDP, distML, "Original", "Blur", "Desv Tipica", "Union")

execute('AS-OCT\im12.jpeg')
plt.show()
