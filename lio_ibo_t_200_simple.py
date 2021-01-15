import numpy as np
import cv2


def lio_ibo(image):
    a, b = 360, 240
    newimage = np.zeros((240, 360), np.ulonglong)
    for x in range(1, 238):
        for y in range(1, 358):
            if image[x][y] < 200:
                newimage[x][y] = 0
            else:
                newimage[x][y] = 255

    newimage = newimage.astype(np.uint8)
    return newimage


img = cv2.imread("frame.jpg", cv2.COLOR_BGR2GRAY)
img = lio_ibo(img)
# run the img through LIO_IBO
cv2.imshow("Window", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
