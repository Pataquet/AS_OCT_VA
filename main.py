import cv2
import numpy as np
import scipy.stats


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

    kernel = np.ones((1,10))
    imgDl = cv2.dilate(imgTh, kernel, iterations = 1)

    return imgOr, imgTh, imgDl

def newImage(image, regionCoords):
    for i in regionCoords:
        image[i[0]][i[1]] = 1
    return image

def cConexas(image, p = False):

    label_image = label(image)
    image_label_overlay = label2rgb(image, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))

    imageComponentesConexas = np.zeros(image.shape)

    ax.imshow(image_label_overlay)
    i = 0
    regions = []
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 1000:
            regions.append(region)

    # To sort the list in place...
    regions.sort(key=lambda x: x.area, reverse=True)

    # To return a new list, use the sorted() built-in function...
    regions = sorted(regions, key=lambda x: x.area, reverse=True)

    regions= regions[0:3]
    for region in regions:
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
    valueComunas = np.zeros((1,columnas))
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
        valueComunas[0][i]= countR1

    plt.figure()
    plt.imshow(out, cmap='gray')
    plt.title('Parts'), plt.xticks([]), plt.yticks([])

    print("Nº total de px: ",totalPX)
    print("Media PX: ", totalPX/columnas)
    return out, valueComunas

def reduceBorders(img):
    filas, columnas = img.shape
    out = np.zeros((filas, columnas))

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('IMG'), plt.xticks([]), plt.yticks([])

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

def mediaTruncada(dist):
    print("MAX: ",max(max(dist)))
    print("MIN: ", min(min(dist)))
    # calculamos los percentile 25% y 75% y hallamos la media recortada
    liminf = scipy.stats.scoreatpercentile(dist, 25)
    limsup = scipy.stats.scoreatpercentile(dist, 75)
    print("El 25% percentil es =", liminf, "y el 75% percentil es =", limsup)
    trimean = scipy.stats.mstats.tmean(dist, (90, 130))
    print("La media recortada es =", trimean)


def execute(th):
    imgOr, imgTh, imgDl  = readImage(th, 'AS-OCT\im12.jpeg')
    imgCc, regiones = cConexas(imgDl)
    imgCombine = combine(imgOr, imgCc)

    # plt.figure()
    # plt.imshow(imgCombine, cmap='gray')
    # plt.title('Combine 1'), plt.xticks([]), plt.yticks([])


    imgCombine[imgCombine > 150] = 255
    imgCombine[imgCombine <= 150] = 0
    imgReduc = reduceBorders(imgCombine)

    imgFin, _ = countPx(imgReduc)
    imgFin, capa1 = countPx(imgFin)
    imgFin, capa2 = countPx(imgFin)
    mediaTruncada(capa1)
    mediaTruncada(capa2)
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
    plt.show()



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


execute(90)