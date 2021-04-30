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


def sortSecond(val):
    return val[1]


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
                    m = image[x][y]
                    p_x = x
                    p_y = y

    return m, p_x, p_y

def binaryimage(image):
    ret, bimg = cv2.threshold(image, 255 * .7, 255, cv2.THRESH_BINARY)
    #print(bimg)
    return bimg

fileName = '3_1.mp4'
cap = cv2.VideoCapture(fileName)  # load the video

all_var = []
intesnity_total = []
size = []
#iii = 50
numberofframes = 0

while (cap.isOpened()):

    ret, img = cap.read()
    if not ret:
        break
    numberofframes = numberofframes + 1
    if numberofframes % 20 == 0:
        #print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = imutils.resize(img, width=320)
        #img = cv2.imread("frame6211.jpg", cv2.COLOR_BGR2GRAY)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        original_img = img

        histr = cv2.calcHist([original_img], [0], None, [256], [0, 256])
        adding_200_255 = 0
        for x in range(len(histr)):
            if x > 199:
                adding_200_255 = adding_200_255 + histr[x][0]
        intesnity_total.append(adding_200_255)
        #cv2.imshow("original", img)
        img = lio_ibo(img)
        g_x_y = img
        #cv2.imshow("g_x_y", img)
        #cv2.imshow("lio_ibo", img)
        img = binaryimage(img)
        #cv2.imshow("binary_image", img)
        #########################################################
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
        #print(nlabels)
        #print(labels)
        print(labels.shape)
        #print(stats)
        #print(stats.shape)
        #print(centroids)
        #print(centroids.shape)
        areas = stats[1:, cv2.CC_STAT_AREA]
        #print(areas)
        #print(areas.shape)


        img = np.zeros((labels.shape), np.uint8)
        img_f = np.zeros((labels.shape), np.uint8)
        amax = max(areas)
        print('Max Area: ', amax)
        areas_include = []

        #print(labels)

        f_k_x_y_images = []
        for i in range(0, nlabels - 1):
            if areas[i] > 40:
                if areas[i] > 0.2*amax:
                    img[labels == i + 1] = 255
                    img_f[labels == i + 1] = 255
                    f_k_x_y_images.append((img_f, areas[i]))
                    img_f = np.zeros((labels.shape), np.uint8)

        f_k_x_y = img

        f_k_x_y_images.sort(key=sortSecond)
        #print(f_k_x_y_images[0][1])
        #print(f_k_x_y_images[1][1])
        #cv2.imshow("f_k_x_y", img)
        #cv2.imshow('labels = 4', img_f)
        #cv2.imshow('f_k_x_y_1', f_k_x_y_images[1][0])
        #cv2.imshow('f_k_x_y_2', f_k_x_y_images[1])
        #print(areas)


        #cv2.imwrite("location_3_fire.jpg", img)


        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, None, None, None, 8, cv2.CV_32S)
        print(stats)
        print(stats.shape)
        print(centroids)
        print(centroids.shape)

        #img = cv2.Canny(img, 100, 200)

        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #contours1, hierarchy1 = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #cnt = contours[1]
        #cv2.drawContours(img, contours1, -1, (155, 0, 0), 1) #drawing countours
        print("Countours")
        #print(contours)
        print(len(contours))
        if len(contours) == 0:
            variance_al = 0
            all_var.append(variance_al)
        #img = cv2.circle(img, (round(centroids[1][0]), round(centroids[1][1])), radius=2, color=(155, 0, 0), thickness=-1)
        #img = cv2.circle(img, (84, 77), radius=2, color=(155, 0, 0), thickness=-1)
        #img = cv2.circle(img, (84, 97), radius=2, color=(155, 0, 0), thickness=-1)
        #print(len(contours[1]))
        #print(len(contours[2]))

        centroids_index = 1
        contours_index = len(contours) - 1

        while centroids_index != len(centroids):
            #iii = iii - 1
            print("###########################################################################################################")
            #print(iii)
            print("centroids_index: ", centroids_index)
            print("contours_index: ", contours_index)
            print("Fk(x,y) area ", f_k_x_y_images[contours_index][1])
            print("############################################################################################################")
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
            mean_al_scaled = sum_a_l_scaled/len(scaledal)
            print('Mean of |a[l]| Scaled: ', mean_al_scaled)

            max_g_x_y_1, xx_1, yy_1 = max_in_image2(g_x_y, f_k_x_y_images[contours_index][0], x, y)
            print('Max in G(x,y): ', max_g_x_y_1, 'x position: ', xx_1, 'y position: ', yy_1)

            max_g_x_y, xx, yy = max_in_image(g_x_y)
            s_3 = max_g_x_y_1 - 5
            #s_3 = 220
            #print('Max in G(x,y): ', max_g_x_y, 'x: ', xx, 'y: ', yy)
            #print(f_k_x_y[xx, yy])
            #if f_k_x_y[xx, yy] > 0:
                #print('True')
                #s_3 = max_g_x_y - 5
            print('S_3: ', s_3)

            if centroids_index == 1:
                pi_img = np.zeros((g_x_y.shape), np.uint8)
                t_img = np.zeros((g_x_y.shape), np.uint8)

            t = 0
            b, a = g_x_y.shape
            for x in range(0, b):
                for y in range(0, a):
                    if g_x_y[x][y] > 0:
                        if f_k_x_y_images[contours_index][0][x][y] > 0:
                            t_img[x][y] = g_x_y[x][y]
                            t = t + 1

            print('Set Cardinality t: ', t)

            pi = 0
            b, a = f_k_x_y.shape
            for x in range(0, b):
                for y in range(0, a):
                    if g_x_y[x][y] > s_3:
                        if f_k_x_y_images[contours_index][0][x][y] > 0:
                            pi_img[x][y] = g_x_y[x][y]
                            pi = pi + 1

            print('Set Cardinality pi: ', pi)

            o = pi / t
            print("Intesity o: ", o)

            #cv2.imshow("pi", pi_img)
            #cv2.imshow("t", t_img)

            mean_al_noabs = sumal_noabs/len(a_l)
            var_num = 0
            for x in range(0, len(a_l)):
                var_num = var_num + ((a_l[x]-mean_al_noabs) ** 2)
            variance_al = var_num/(len(a_l))
            print("Mean a[l]:", mean_al_noabs, "Variance a[l]: ", variance_al)
            a1 = variance_al/t
            print("variance/area: ", a1)

            #cnt = contours[contours_index]
            #rect = cv2.minAreaRect(cnt)
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #img = cv2.drawContours(img, [box], 0, (155, 0, 0), 1)

            #cv2.imshow("Done", img)

            #plot1 = plt.figure(centroids_index)

            #fig, axs = plt.subplots(3)
            #fig.suptitle('Index: %g' %centroids_index)
            #axs[0].set_title('Original Signal')
            #axs[0].plot(new_x, new_y)
            #axs[1].set_title('High Pass')
            #axs[1].plot(a_l)
            #axs[2].set_title('Low Pass')
            #axs[2].plot(d_l)

            #print(original_img.shape)
            #b, a = original_img.shape
            #values = []
            #for x in range(0, b):
                #for y in range(0, a):
                    #values.append(original_img[x][y])
            #axs[3].hist(values)
            #print(values)
            all_var.append(variance_al)
            size.append(t)

            rho = 0.75*mean_al + 0.25*o
            print('r: ', rho)
            if rho > 0.275:
                print('Fire!!!')
            else:
                print('Non-Fire')
            ####################################################################################################################
            centroids_index = centroids_index + 1
            contours_index = contours_index - 1
            print("###########################################################################################################")
            break
        #if iii == 0:
            #break


#plt.show()
#fig, ax = plt.subplots()
fig, axs = plt.subplots(3)
axs[0].plot(intesnity_total, label="line1")
axs[1].plot(all_var, label="line2")
axs[2].plot(size, label="line3")
axs[1].set_ylim([0, 13])
axs[2].set_ylim([0, 2500])

#plt.xlabel('frame')
#plt.ylabel('variance')
#plt.title('Variance Of High Pass At Each Frame')
#ax.add_patch(Rectangle((18, 0), 5, 5, edgecolor='red', fill=False))
#plt.show()
#plt.plot(intesnity_total)
#plt.xlabel('frame')
#plt.ylabel('total # pixels from 200-255')
#plt.title('Total # Of Pixels (200-255) at Each Frame')
#ax.add_patch(Rectangle((17, 5000), 5, 1500, edgecolor='red', fill=False))
plt.show()
print(all_var)
print(len(all_var))
print(intesnity_total)
cv2.waitKey(0)
cv2.destroyAllWindows()
