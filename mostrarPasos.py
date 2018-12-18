import matplotlib.pyplot as plt


def showImage6(img1, img2,  img3, img4, img5, img6, title1 = None, title2 = None, title3 = None, title4 = None, title5 = None, title6 = None):
    plt.figure()
    plt.subplot(321), plt.imshow(img1, cmap='gray')
    plt.title(title1), plt.xticks([]), plt.yticks([])

    plt.subplot(322), plt.imshow(img2, cmap='gray')
    plt.title(title2), plt.xticks([]), plt.yticks([])

    plt.subplot(323), plt.imshow(img3, cmap='gray')
    plt.title(title3), plt.xticks([]), plt.yticks([])

    plt.subplot(324), plt.imshow(img4, cmap='gray')
    plt.title(title4), plt.xticks([]), plt.yticks([])

    plt.subplot(325), plt.imshow(img5, cmap='gray')
    plt.title(title5), plt.xticks([]), plt.yticks([])

    plt.subplot(326), plt.imshow(img6, cmap='gray')
    plt.title(title6), plt.xticks([]), plt.yticks([])

def showImage3(img1, img2,  img3, title1 = None, title2 = None, title3 = None):
    plt.figure()
    plt.subplot(131), plt.imshow(img1, cmap='gray')
    plt.title(title1), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(img2, cmap='gray')
    plt.title(title2), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(img3, cmap='gray')
    plt.title(title3), plt.xticks([]), plt.yticks([])


def showImage4(img1, img2,  img3, img4, title1 = None, title2 = None, title3 = None, title4 = None):
    plt.figure()
    plt.subplot(221), plt.imshow(img1, cmap='gray')
    plt.title(title1), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(img2, cmap='gray')
    plt.title(title2), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(img3, cmap='gray')
    plt.title(title3), plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(img4, cmap='gray')
    plt.title(title4), plt.xticks([]), plt.yticks([])

def showStadistics(st1, st2,  st3, st4, title1 = None, title2 = None, title3 = None, title4 = None, title5 = None):

    plt.figure()
    plt.plot(st1)
    plt.plot(st2)
    plt.plot(st3)
    plt.plot(st4)
    plt.title(title5)


    plt.figure()
    plt.subplot(221), plt.plot(st1)
    plt.title(title1)

    plt.subplot(222), plt.plot(st2)
    plt.title(title2)

    plt.subplot(223), plt.plot(st3)
    plt.title(title3)

    plt.subplot(224), plt.plot(st4)
    plt.title(title4)