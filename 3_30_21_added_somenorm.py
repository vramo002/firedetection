import numpy as np
import cv2
import pywt
import pywt.data
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from skimage.draw import line
from sklearn.preprocessing import minmax_scale


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

def max_in_image2(image, image2, xs, ys):
    b, a = image.shape
    m = 0
    p_x = 0
    p_y = 0
    for x in range(0, b):
        for y in range(0, a):
            if m < image[x][y]: #found potental max
                if image2[x][y] > 0: #check if in fkxy
                    for z in range(0, len(xs)): #finding x and y in the x and y of the countors
                        if xs[z] == x & ys[z] == y:
                            print("3")
                            m = image[x][y]
                            p_x = x
                            p_y = y

    return m, p_x, p_y

def binaryimage(image):
    ret, bimg = cv2.threshold(image, 255 * .7, 255, cv2.THRESH_BINARY)
    #print(bimg)
    return bimg


img = cv2.imread("test4.jpg", cv2.COLOR_BGR2GRAY)
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
#print(labels)

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
    sumal_noabs = sum(a_l)
    sumdl_noabs = sum(d_l)

    ###############################################
    #high_using = [0 for i in range(0, len(angles))]
    #low = np.convolve(new_y, g, 'same')
    #high = np.convolve(new_y, h, 'same')
    #for i in range(0, len(low)):
        #high_using[i] = new_y[i] - low[i]
    ###############################################

    a_l_abs = [abs(l) for l in a_l]
    d_l_abs = [abs(h) for h in d_l]

    sum_a_l = sum(a_l_abs)
    sum_d_l = sum(d_l_abs)
    sum_new_y = sum(new_y)
    print('Sum of |d[l]|: ', sum_d_l, 'Sum of |a[l]|: ', sum_a_l, 'Sum of OG Signal: ', sum_new_y)
    mean_dl = sum_d_l/len(d_l)
    mean_al = sum_a_l/len(a_l)
    print('Mean of |d[l]|: ', mean_dl, 'Mean of |a[l]|: ', mean_al)
    beta = sum_a_l / sum_d_l
    print('Beta: High/Low', beta)
    beta_norm = ((beta - mean_al) / (mean_dl - mean_al))
    print('Beta Normalize: ', beta_norm)
    h_og = sum_a_l/sum_new_y
    l_og = sum_d_l/sum_new_y
    print('High Pass/Original Signal: ', h_og, 'Low Pass/Original Signal: ', l_og)

    scaledal = minmax_scale(a_l_abs)
    scaleddl = minmax_scale(d_l_abs)
    sum_a_l_scaled = sum(scaledal)
    sum_d_l_scaled = sum(scaleddl)
    beta2 = sum_a_l_scaled / sum_d_l_scaled
    print('Beta when Scaled: High/Low', beta2)

    #max_g_x_y, xx, yy = max_in_image2(g_x_y, f_k_x_y, x, y)
    max_g_x_y, xx, yy = max_in_image(g_x_y)
    s_3 = max_g_x_y - 5
    #print('Max in G(x,y): ', max_g_x_y, 'x: ', xx, 'y: ', yy)
    #print(f_k_x_y[xx, yy])
    #if f_k_x_y[xx, yy] > 0:
        #print('True')
        #s_3 = max_g_x_y - 5
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










#plot1 = plt.figure(1)
#plt.plot(a_l)
#plot2 = plt.figure(2)
#plt.plot(d_l)




#print(len(angles))

#angles.sort()
#plt.plot(new_x, new_y)
#plt.show()


#img = cv2.circle(img, center, radius=2, color=(155, 0, 0), thickness=-1)
#img1 = np.zeros((labels.shape), np.uint8) #blank image
#img = cv2.drawContours(img1, contours, -1, (155, 0, 0), 1) #drawing countours

#cnt = contours[0]
#leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
#rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
#topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
#bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
#distance = math.sqrt(((leftmost[0]-centroids[2][0])**2) + ((leftmost[1]-centroids[2][1])**2))
#distance1 = math.sqrt(((rightmost[0]-centroids[2][0])**2) + ((rightmost[1]-centroids[2][1])**2))
#distance2 = math.sqrt(((topmost[0]-centroids[2][0])**2) + ((topmost[1]-centroids[2][1])**2))
#distance3 = math.sqrt(((bottommost[0]-centroids[2][0])**2) + ((bottommost[1]-centroids[2][1])**2))

#print(leftmost)
#print(rightmost)
#print(topmost)
#print(bottommost)
#print(distance)
#print(distance1)
#print(distance2)
#print(distance3)


#(x, y), radius = cv2.minEnclosingCircle(cnt)
#center = (int(x), int(y))
#radius = int(radius)
#cv2.circle(img, (round(centroids[2][0]), round(centroids[2][1])), round(distance1), (155, 0, 0), 1)
#print(centroids[2][0])
#print(centroids[2][1])
#img = cv2.circle(img, (round(centroids[2][0]), round(centroids[2][1])), radius=2, color=(155, 0, 0), thickness=-1)

#pointsx = [0 for i in range(0, 359)]
#pointsy = [0 for i in range(0, 359)]
#for i in range(0, 359):
    #pointsx[i] = distance1 * math.cos(i*math.pi/180) + centroids[2][0]
    #pointsy[i] = distance1 * math.sin(i*math.pi/180) + centroids[2][1]

#for i in range(0, 359):
    #img = cv2.circle(img, (round(pointsx[i]), round(pointsy[i])), radius=1, color=(155, 0, 0), thickness=-1)

#rr, cc = line(round(pointsx[90]), round(pointsy[90]), round(centroids[2][0]), round(centroids[2][1]))
#img[cc, rr] = 155
#print(rr)
#print(cc)

#cnt = contours[1]
#leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
#rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
#topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
#bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
#distance = math.sqrt(((leftmost[0]-centroids[1][0])**2) + ((leftmost[1]-centroids[1][1])**2))
#distance1 = math.sqrt(((rightmost[0]-centroids[1][0])**2) + ((rightmost[1]-centroids[1][1])**2))
#distance2 = math.sqrt(((topmost[0]-centroids[1][0])**2) + ((topmost[1]-centroids[1][1])**2))
#distance3 = math.sqrt(((bottommost[0]-centroids[1][0])**2) + ((bottommost[1]-centroids[1][1])**2))

#print(leftmost)
#print(rightmost)
#print(topmost)
#print(bottommost)
#print(distance)
#print(distance1)
#print(distance2)
#print(distance3)

#(x, y), radius = cv2.minEnclosingCircle(cnt)
#center = (int(x), int(y))
#radius = int(radius)
#cv2.circle(img, (round(centroids[1][0]), round(centroids[1][1])), round(distance1), (155, 0, 0), 1)
#print(center)
#img = cv2.circle(img, (round(centroids[1][0]), round(centroids[1][1])), radius=2, color=(155, 0, 0), thickness=-1)

#pointsx = [0 for i in range(0, 359)]
#pointsy = [0 for i in range(0, 359)]
#for i in range(0, 359):
    #pointsx[i] = distance1 * math.cos(i*math.pi/180) + centroids[1][0]
    #pointsy[i] = distance1 * math.sin(i * math.pi / 180) + centroids[1][1]

#for i in range(0, 359):
    #img = cv2.circle(img, (round(pointsx[i]), round(pointsy[i])), radius=1, color=(155, 0, 0), thickness=-1)


#print(centroids[1])
#cx = round(centroids[1][0])
#cy = round(centroids[1][1])
#print(cx)
#print(cy)
#print(centroids[2])
#cx = round(centroids[2][0])
#cy = round(centroids[2][1])
#print(cx)
#print(cy)

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
#cnt = contours[0]
#rect = cv2.minAreaRect(cnt)
#box = cv2.boxPoints(rect)
#box = np.int0(box)
#img = cv2.drawContours(img, [box], 0, (155, 0, 0), 1)

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

#plot3 = plt.figure(3)
#plt.plot(new_x, new_y)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
