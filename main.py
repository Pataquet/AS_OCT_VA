import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('AS-OCT\im3.jpeg', 0)
# plt.title('Original Image')
# plt.show()

def readCol(umbral, img):
    print(type(img[0][0]))
    rows, cols = img.shape
    listMax = np.zeros((cols,rows), dtype=np.uint8)
    print(type(listMax[0][0]))

    for i in range(cols):
        for j in range(rows):
            listMax[i][j] = int(img[j][i])

    plt.imshow(listMax, cmap='gray')
    plt.title('Original Image')
    plt.show()

def a (thigh, img):
    rows, cols = img.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    mayorThigh = (dst > thigh)

    outImage = np.zeros(dst.shape)
    filas, columnas = dst.shape


    # u.mostrarImagenes(mayorThigh, mayorTlow)
    pixelsVisitados = []
    for r in range(1, filas - 1):
        for c in range(1, columnas - 1):
            #ignora todos los valores inferiores a tlow
            if mayorThigh[r, c] != 1:
                continue  # Not a weak pixel

            # Get 3x3 patch
            localPatch = mayorThigh[r - 1:r + 2, c - 1:c + 2]
            patchMax = localPatch.max()
            #si uno de los vecinos del px tiene valor 2 se añade a la matriz para comprobar mas adelante
            # si tienes mas vecinos superiores a tlow
            if patchMax == 1:
                pixelsVisitados.append((r, c))
                outImage[r, c] = 1

    plt.imshow(outImage, cmap='gray')
    plt.title('Original Image')
    plt.show()


a(150, img)

# kernel = np.ones((1,10))
# median = cv2.medianBlur(img,9)
#
# dilate = cv2.dilate(median,kernel,iterations = 1)
# edges = cv2.Canny(dilate,100,200)

# plt.subplot(221),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(222),plt.imshow(median,cmap = 'gray')
# plt.title('Median Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(223),plt.imshow(edges,cmap = 'gray')
# plt.title('Edges Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(224),plt.imshow(dilate,cmap = 'gray')
# plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])
#
#
# plt.show()


# def find_if_close(cnt1,cnt2):
#     row1,row2 = cnt1.shape[0],cnt2.shape[0]
#     for i in range(row1):
#         for j in range(row2):
#             dist = np.linalg.norm(cnt1[i]-cnt2[j])
#             if abs(dist) < 50 :
#                 return True
#             elif i==row1-1 and j==row2-1:
#                 return False
#
# img = cv2.imread('AS-OCT\im3.jpeg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(gray,127,255,0)
# _, contours, hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,2)
#
#
#
# cv2.drawContours(img,contours,-1,(0,255,0),2)
# cv2.drawContours(edges,contours,-1,255,-1)
#
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])
#
# LENGTH = len(contours)
# status = np.zeros((LENGTH,1))
#
# for i,cnt1 in enumerate(contours):
#     x = i
#     if i != LENGTH-1:
#         for j,cnt2 in enumerate(contours[i+1:]):
#             x = x+1
#             dist = find_if_close(cnt1,cnt2)
#             if dist == True:
#                 val = min(status[i],status[x])
#                 status[x] = status[i] = val
#             else:
#                 if status[x]==status[i]:
#                     status[x] = i+1
#
# unified = []
# maximum = int(status.max())+1
# for i in range(maximum):
#     pos = np.where(status==i)[0]
#     if pos.size != 0:
#         cont = np.vstack(contours[i] for i in pos)
#         hull = cv2.convexHull(cont)
#         unified.append(hull)
#
# cv2.drawContours(img,unified,-1,(0,255,0),2)
# cv2.drawContours(thresh,unified,-1,255,-1)
#
# plt.subplot(122),plt.imshow(img,cmap = 'gray')
# plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])

# plt.show()
#
#

