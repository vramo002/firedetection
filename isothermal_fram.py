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

img1 = cv2.imread("frametest#1.jpg")
img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 100, 100])
upper_red = np.array([179, 255, 255])

mask = cv2.inRange(img, lower_red, upper_red)

res = cv2.bitwise_and(img1, img1, mask=mask)

cv2.imshow("original", img)
cv2.imshow('mask', mask)
cv2.imshow('res', res)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
