import numpy as np
import cv2


def lio_ibo(image):
    b, a = image.shape
    newimage = [[0 for i in range(a)] for j in range(b)]
    #newimage = np.zeros((240, 360), np.ulonglong)
    for x in range(1, b-2):
        for y in range(1, a-2):
            newimage[x][y] = (int(image[x - 1][y - 1]) * int(image[x - 1][y]) * int(image[x - 1][y + 1]) *
                              int(image[x][y - 1]) * int(image[x][y]) * int(image[x][y + 1]) * int(image[x + 1][y - 1])
                              * int(image[x + 1][y]) * int(image[x + 1][y + 1]))

    m = 0
    for x in range(1, b-2):
        for y in range(1, a-2):
            if m < newimage[x][y]:
                m = newimage[x][y]

    #print(m)

    for x in range(1, b-2):
        for y in range(1, a-2):
            newimage[x][y] = newimage[x][y]/m

    m = 0
    for x in range(1, b-2):
        for y in range(1, a-2):
            if m < newimage[x][y]:
                m = newimage[x][y]

    #print(m)

    for x in range(1, b-2):
        for y in range(1, a-2):
            newimage[x][y] = newimage[x][y]*255


    #newimage = newimage.astype(np.uint8)
    return np.uint8(newimage)


def mat(image):
    a, b = 360, 240
    newimage = [[0 for i in range(a)] for j in range(b)]

    for x in range(1, 238):
        for y in range(1, 358):
            if image[x][y] < 128:
                newimage[x][y] = 255
            else:
                newimage[x][y] = 0

    return np.uint8(newimage)


def gmax(image):
    b, a = image.shape
    m = 0
    for x in range(1, b-2):
        for y in range(1, a-2):
            if m < image[x][y]:
                m = image[x][y]

    return m


def binaryimage(image):
    ret, bimg = cv2.threshold(image, 255 * .7, 255, cv2.THRESH_BINARY)
    #print(bimg)
    return bimg


img = cv2.imread("test3.jpg", cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = lio_ibo(img)
cv2.imshow("lio_ibo", img)
img = binaryimage(img)
cv2.imshow("binary_image", img)
#########################################################
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
print(nlabels)
print(labels)
print(labels.shape)
print(stats)
print(stats.shape)
print(centroids)
print(centroids.shape)
areas = stats[1:, cv2.CC_STAT_AREA]
print(areas)
print(areas.shape)

img = np.zeros((labels.shape), np.uint8)
amax = max(areas)
print(amax)

for i in range(0, nlabels - 1):
    if areas[i] > 40:
        if areas[i] > 0.2*amax:
            img[labels == i + 1] = 255

#status = cv2.imwrite('img1_1order_ibo.jpg', img)
#img = mat(img)
#img = lio_ibo(img)
#status = cv2.imwrite('img1_2order_ibo.jpg', img)
#img = mat(img)
#status = cv2.imwrite('img1_segmetation_mat.jpg', img)
#img = lio_ibo(img)


# run the img through LIO_IBO
cv2.imshow("Done", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
