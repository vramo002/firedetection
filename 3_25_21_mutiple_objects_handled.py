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

def max_in_image(image):
    b, a = image.shape
    m = 0
    p_x = 0
    p_y = 0
    for x in range(0, b):
        for y in range(0, a):
            if m < image[x][y]:
                m = image[x][y]
                p_x = x
                p_y = y

    return m, p_x, p_y

def binaryimage(image):
    ret, bimg = cv2.threshold(image, 255 * .7, 255, cv2.THRESH_BINARY)
    #print(bimg)
    return bimg


img = cv2.imread("test3.jpg", cv2.COLOR_BGR2GRAY)
original_img = img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("original", img)
img = lio_ibo(img)
g_x_y = img
cv2.imshow("g_x_y", img)
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

f_k_x_y = img

cv2.imshow("f_k_x_y", img)
#cv2.imwrite("location_3_fire.jpg", img)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
print(stats)
print(stats.shape)
print(centroids)
print(centroids.shape)

#img = cv2.Canny(img, 100, 200)

contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#cnt = contours[1]
#cv2.drawContours(img, contours, -1, (155, 0, 0), 2) #drawing countours
print("Countours")
#print(contours)
print(len(contours))
#img = cv2.circle(img, (round(centroids[1][0]), round(centroids[1][1])), radius=2, color=(155, 0, 0), thickness=-1)
#img = cv2.circle(img, (84, 77), radius=2, color=(155, 0, 0), thickness=-1)
#img = cv2.circle(img, (84, 97), radius=2, color=(155, 0, 0), thickness=-1)
#print(len(contours[1]))
#print(len(contours[2]))

centroids_index = 1
contours_index = len(contours) - 1

while centroids_index != len(centroids):
    print("centroids_index: ", centroids_index)
    print("contours_index: ", contours_index)
    ####################################################################################################################
    xy = contours[contours_index]
    x = [0 for i in range(0, len(xy))]
    y = [0 for i in range(0, len(xy))]
    for i in range(0, len(contours[contours_index])):
        x[i] = xy[i][0][0]
    for i in range(0, len(contours[contours_index])):
        y[i] = xy[i][0][1]

    cx = round(centroids[centroids_index][0])
    cy = round(centroids[centroids_index][1])

    distance = [0 for i in range(0, len(xy))]
    for i in range(0, len(xy)):
        distance[i] = math.sqrt(((x[i] - cx) ** 2) + ((y[i] - cy) ** 2))

    max_distance = max(distance)
    max_location = distance.index(max(distance))

    angle = [0 for i in range(0, len(xy))]  # for now it does not have 0 to 360 degrees
    for i in range(0, len(angle)):
        angle[i] = i

    for i in range(0, len(xy)):
        x[i] = x[i] - cx

    for i in range(0, len(xy)):
        y[i] = y[i] - cy

    angles = [0 for i in range(0, len(xy))]

    for i in range(0, len(angles)):
        if x[i] > 0 and y[i] >= 0:
            angles[i] = math.atan2(y[i], x[i]) * (180 / math.pi)
        elif x[i] < 0 and y[i] >= 0:
            angles[i] = math.atan2(y[i], x[i]) * (180 / math.pi)
        elif x[i] < 0 and y[i] < 0:
            angles[i] = 360 + (math.atan2(y[i], x[i]) * (180 / math.pi))
        elif x[i] > 0 and y[i] < 0:
            angles[i] = 360 + (math.atan2(y[i], x[i]) * (180 / math.pi))
        elif x[i] == 0 and y[i] >= 0:
            angles[i] = 90
        elif x[i] == 0 and y[i] < 0:
            angles[i] = 270

    d = np.array([angles])
    g = angles[max_location]

    for i in range(0, len(angles)):
        angles[i] = angles[i] - g
        if angles[i] < 0:
            angles[i] = angles[i] + 360

    d = np.array([angles])

    xy_pairs = list(zip(angles, distance))
    xy_pairs.sort()

    new_x = [0 for i in range(0, len(angles))]
    new_y = [0 for i in range(0, len(angles))]
    for i in range(0, len(contours[contours_index])):
        new_x[i] = xy_pairs[i][0]
    for i in range(0, len(contours[contours_index])):
        new_y[i] = xy_pairs[i][1]

    h = [-1/4, 1/2, -1/4]
    g = [1/4, 1/2, 1/4]

    a_l = np.convolve(new_y, h, 'valid')  # high
    d_l = np.convolve(new_y, g, 'valid')  # low

    a_l_abs = [abs(l) for l in a_l]
    d_l_abs = [abs(h) for h in d_l]

    sum_a_l = sum(a_l_abs)
    sum_d_l = sum(d_l_abs)
    print('Sum of |d[l]|: ', sum_d_l, 'Sum of |a[l]|: ', sum_a_l)
    beta = sum_a_l / sum_d_l
    print('Beta: ', beta)

    max_g_x_y, xx, yy = max_in_image(g_x_y)
    s_3 = 0
    print('Max in G(x,y): ', max_g_x_y, 'x: ', xx, 'y: ', yy)
    print(f_k_x_y[xx, yy])
    if f_k_x_y[xx, yy] > 0:
        print('True')
        s_3 = max_g_x_y - 35
    print('S_3: ', s_3)

    pi_img = np.zeros((g_x_y.shape), np.uint8)
    t_img = np.zeros((g_x_y.shape), np.uint8)

    t = 0
    b, a = g_x_y.shape
    for x in range(0, b):
        for y in range(0, a):
            if g_x_y[x][y] > 0:
                if f_k_x_y[x][y] > 0:
                    t_img[x][y] = g_x_y[x][y]
                    t = t + 1

    print('Set Cardinality t: ', t)

    pi = 0
    b, a = f_k_x_y.shape
    for x in range(0, b):
        for y in range(0, a):
            # qprint(g_x_y[x][y])
            if g_x_y[x][y] > s_3:
                if f_k_x_y[x][y] > 0:
                    pi_img[x][y] = g_x_y[x][y]
                    pi = pi + 1

    print('Set Cardinality pi: ', pi)

    o = pi / t
    print("Intesity o: ", o)

    cv2.imshow("pi", pi_img)
    cv2.imshow("t", t_img)

    #cnt = contours[contours_index]
    #rect = cv2.minAreaRect(cnt)
    #box = cv2.boxPoints(rect)
    #box = np.int0(box)
    #img = cv2.drawContours(img, [box], 0, (155, 0, 0), 1)

    cv2.imshow("Done", img)

    #plot1 = plt.figure(centroids_index)
    fig, axs = plt.subplots(3)
    fig.suptitle('Index: %g' %centroids_index)
    axs[0].set_title('Original Signal')
    axs[0].plot(new_x, new_y)
    axs[1].set_title('High Pass')
    axs[1].plot(a_l)
    axs[2].set_title('Low Pass')
    axs[2].plot(d_l)
    ####################################################################################################################
    centroids_index = centroids_index + 1
    contours_index = contours_index - 1

    
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
