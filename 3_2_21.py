import numpy as np
import cv2
import pywt
import pywt.data
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from skimage.draw import line


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


img = cv2.imread("FireWhirlOn_IR_camera.jpg", cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = lio_ibo(img)
cv2.imshow("lio_ibo", img)
img = binaryimage(img)
cv2.imshow("binary_image", img)
#########################################################
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
#print(nlabels)
#print(labels)
#print(labels.shape)
#print(stats)
#print(stats.shape)
#print(centroids)
#print(centroids.shape)
areas = stats[1:, cv2.CC_STAT_AREA]
#print(areas)
#print(areas.shape)

img = np.zeros((labels.shape), np.uint8)
amax = max(areas)
print(amax)

for i in range(0, nlabels - 1):
    if areas[i] > 40:
        if areas[i] > 0.2*amax:
            img[labels == i + 1] = 255

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
print(stats)
print(stats.shape)
print(centroids)
print(centroids.shape)

#img = cv2.Canny(img, 100, 200)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cnt = contours[1]
#cv2.drawContours(img, contours, -1, (155, 0, 0), 2) #drawing countours
print(len(contours))
print(len(contours[0]))
print(len(contours[1]))
#print(len(contours[2]))

xy = contours[0]
x = [0 for i in range(0, len(xy))]
y = [0 for i in range(0, len(xy))]
for i in range(0, len(contours[0])):
    x[i] = xy[i][0][0]
    #print(x)
for i in range(0, len(contours[0])):
    y[i] = xy[i][0][1]

print(x)
print(y)
#print(xy)

#print(centroids[1])
cx = round(centroids[2][0])
cy = round(centroids[2][1])
print(cx)
print(cy)

distance = [0 for i in range(0, len(xy))]
for i in range(0, len(xy)):
    distance[i] = math.sqrt(((x[i]-cx)**2) + ((y[i]-cy)**2))

#print(distance)

angle = [0 for i in range(0, len(xy))] #for now it does not have 0 to 360 degrees
for i in range(0, len(angle)):
    angle[i] = i

#print(angle)

for i in range(0, len(xy)):
    x[i] = x[i] - cx

for i in range(0, len(xy)):
    y[i] = y[i] - cy

print(x)
print(y)

angles = [0 for i in range(0, len(xy))]
print(y[0], x[0])
print(y[1], x[1])
print(math.atan2(y[0], x[0]) * (180/math.pi))
print(math.atan2(y[1], x[1]) * (180/math.pi))
for i in range(0, len(angles)):
    if x[i] > 0 and y[i] >= 0:
        angles[i] = math.atan2(y[i], x[i]) * (180/math.pi)
    elif x[i] < 0 and y[i] >= 0:
        angles[i] = math.atan2(y[i], x[i]) * (180/math.pi)
    elif x[i] < 0 and y[i] < 0:
        angles[i] = 360 + (math.atan2(y[i], x[i]) * (180/math.pi))
    elif x[i] > 0 and y[i] < 0:
        angles[i] = 360 + (math.atan2(y[i], x[i]) * (180/math.pi))
    elif x[i] == 0 and y[i] >= 0:
        angles[i] = 90
    elif x[i] == 0 and y[i] < 0:
        angles[i] = 270

print(angles)
#d = np.array([angles])
#print(d)
#n_angles = preprocessing.normalize(d)
#print(n_angles)

#for i in range(0, len(n_angles)):
    #n_angles[i] = n_angles[i] * 3600

#print(n_angles)

#print(len(angles))


plt.plot(angles, distance)
plt.show()


#img = cv2.circle(img, center, radius=2, color=(155, 0, 0), thickness=-1)
img1 = np.zeros((labels.shape), np.uint8) #blank image
img = cv2.drawContours(img1, contours, -1, (155, 0, 0), 1) #drawing countours

cnt = contours[0]
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
distance = math.sqrt(((leftmost[0]-centroids[2][0])**2) + ((leftmost[1]-centroids[2][1])**2))
distance1 = math.sqrt(((rightmost[0]-centroids[2][0])**2) + ((rightmost[1]-centroids[2][1])**2))
distance2 = math.sqrt(((topmost[0]-centroids[2][0])**2) + ((topmost[1]-centroids[2][1])**2))
distance3 = math.sqrt(((bottommost[0]-centroids[2][0])**2) + ((bottommost[1]-centroids[2][1])**2))

print(leftmost)
print(rightmost)
print(topmost)
print(bottommost)
print(distance)
print(distance1)
print(distance2)
print(distance3)


(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(img, (round(centroids[2][0]), round(centroids[2][1])), round(distance1), (155, 0, 0), 1)
print(centroids[2][0])
print(centroids[2][1])
img = cv2.circle(img, (round(centroids[2][0]), round(centroids[2][1])), radius=2, color=(155, 0, 0), thickness=-1)

pointsx = [0 for i in range(0, 359)]
pointsy = [0 for i in range(0, 359)]
for i in range(0, 359):
    pointsx[i] = distance1 * math.cos(i*math.pi/180) + centroids[2][0]
    pointsy[i] = distance1 * math.sin(i*math.pi/180) + centroids[2][1]

for i in range(0, 359):
    img = cv2.circle(img, (round(pointsx[i]), round(pointsy[i])), radius=1, color=(155, 0, 0), thickness=-1)

rr, cc = line(round(pointsx[90]), round(pointsy[90]), round(centroids[2][0]), round(centroids[2][1]))
img[cc, rr] = 155
print(rr)
print(cc)

cnt = contours[1]
leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
distance = math.sqrt(((leftmost[0]-centroids[1][0])**2) + ((leftmost[1]-centroids[1][1])**2))
distance1 = math.sqrt(((rightmost[0]-centroids[1][0])**2) + ((rightmost[1]-centroids[1][1])**2))
distance2 = math.sqrt(((topmost[0]-centroids[1][0])**2) + ((topmost[1]-centroids[1][1])**2))
distance3 = math.sqrt(((bottommost[0]-centroids[1][0])**2) + ((bottommost[1]-centroids[1][1])**2))

print(leftmost)
print(rightmost)
print(topmost)
print(bottommost)
print(distance)
print(distance1)
print(distance2)
print(distance3)

(x, y), radius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(img, (round(centroids[1][0]), round(centroids[1][1])), round(distance1), (155, 0, 0), 1)
print(center)
img = cv2.circle(img, (round(centroids[1][0]), round(centroids[1][1])), radius=2, color=(155, 0, 0), thickness=-1)

#pointsx = [0 for i in range(0, 359)]
#pointsy = [0 for i in range(0, 359)]
#for i in range(0, 359):
    #pointsx[i] = distance1 * math.cos(i*math.pi/180) + centroids[1][0]
    #pointsy[i] = distance1 * math.sin(i * math.pi / 180) + centroids[1][1]

#for i in range(0, 359):
    #img = cv2.circle(img, (round(pointsx[i]), round(pointsy[i])), radius=1, color=(155, 0, 0), thickness=-1)


print(centroids[1])
cx = round(centroids[1][0])
cy = round(centroids[1][1])
print(cx)
print(cy)
print(centroids[2])
cx = round(centroids[2][0])
cy = round(centroids[2][1])
print(cx)
print(cy)

#cnt = contours[1]


#print(cnt)
#print(cnt.shape)
#leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
#rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
#topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
#bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
#print(leftmost)
#rint(rightmost)
#print(topmost)
#print(bottommost)

#mask = np.zeros(img.shape, np.uint8)
#cv2.drawContours(mask, [cnt], 0, 255, -1)
#pixelpoints = cv2.findNonZero(mask)

#mean_val = cv2.mean(img, mask=mask)
#print(mean_val)

#img = cv2.circle(img, leftmost, radius=2, color=(155, 0, 0), thickness=-1)
#img = cv2.circle(img, rightmost, radius=2, color=(155, 0, 0), thickness=-1)
#img = cv2.circle(img, topmost, radius=2, color=(155, 0, 0), thickness=-1)
#img = cv2.circle(img, bottommost, radius=2, color=(155, 0, 0), thickness=-1)
#img = cv2.circle(img, (271, 412), radius=2, color=(155, 0, 0), thickness=-1)

#mask = np.zeros(img.shape, np.uint8)
#cv2.drawContours(mask, [cnt], 0, 255, -1)
#pixelpoints = cv2.findNonZero(mask)


#print(pixelpoints)
#rect = cv2.minAreaRect(cnt)
#box = cv2.boxPoints(rect)
#box = np.int0(box)
#img = cv2.drawContours(img, [box], 0, (155, 0, 0), 2)

#print(contours.shape)
#img = cv2.Canny(img, 100, 200) #do the numbers even matter canny edge detection

#status = cv2.imwrite('img1_1order_ibo.jpg', img)
#img = mat(img)
#img = lio_ibo(img)
#status = cv2.imwrite('img1_2order_ibo.jpg', img)
#img = mat(img)
#status = cv2.imwrite('img1_segmetation_mat.jpg', img)
#img = lio_ibo(img)


# run the img through LIO_IBO
cv2.imshow("Done", img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
