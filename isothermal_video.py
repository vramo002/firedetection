import numpy as np
import cv2
import pywt
import pywt.data
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math
from sklearn import preprocessing
from skimage.draw import line
from sklearn.preprocessing import minmax_scale
import imutils


fileName = 'test5_5_21.mp4'
cap = cv2.VideoCapture(fileName)

fps = cap.get(cv2.CAP_PROP_FPS)

lower_red = np.array([0, 100, 100])
upper_red = np.array([179, 255, 255])

numberofframes = 0
foundred = []

while (cap.isOpened()):

    ret, img = cap.read()
    if not ret:
        break
    numberofframes = numberofframes + 1
    if numberofframes % 20 == 0:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = imutils.resize(img, width=320)

        mask = cv2.inRange(img, lower_red, upper_red)
        res = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(gray) != 0:
            foundred.append(1)
        if cv2.countNonZero(gray) == 0:
            foundred.append(0)
        #cv2.imshow("original", img)
        #cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        cv2.waitKey(int((1 / int(fps)) * 1000))

plt.plot(foundred, '-', linewidth=1)
plt.ylim([0, 2])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
