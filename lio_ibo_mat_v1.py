import numpy as np
import cv2


def lio_ibo(image):
    a, b = 360, 240
    newimage = [[0 for i in range(a)] for j in range(b)]
    #newimage = np.zeros((240, 360), np.ulonglong)
    for x in range(1, 238):
        for y in range(1, 358):
            newimage[x][y] = (int(image[x - 1][y - 1]) * int(image[x - 1][y]) * int(image[x - 1][y + 1]) *
                              int(image[x][y - 1]) * int(image[x][y]) * int(image[x][y + 1]) * int(image[x + 1][y - 1])
                              * int(image[x + 1][y]) * int(image[x + 1][y + 1]))

    max = 0
    for x in range(1, 238):
        for y in range(1, 358):
            if max < newimage[x][y]:
                max = newimage[x][y]

    print(max)

    for x in range(1, 238):
        for y in range(1, 358):
            newimage[x][y] = newimage[x][y]/max

    max = 0
    for x in range(1, 238):
        for y in range(1, 358):
            if max < newimage[x][y]:
                max = newimage[x][y]

    print(max)

    for x in range(1, 238):
        for y in range(1, 358):
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


img = cv2.imread("frame.jpg", cv2.COLOR_BGR2GRAY)
img = lio_ibo(img)
img = mat(img)
#img = lio_ibo(img)
#img = mat(img)
#img = lio_ibo(img)

# run the img through LIO_IBO
cv2.imshow("Window", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
