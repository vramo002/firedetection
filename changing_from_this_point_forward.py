xy = contours[0]
x = [0 for i in range(0, len(xy))]
y = [0 for i in range(0, len(xy))]
for i in range(0, len(contours[0])):
    x[i] = xy[i][0][0]
    #print(x)
for i in range(0, len(contours[0])):
    y[i] = xy[i][0][1]
print("X Values")
print(x)
print("Y Values")
print(y)
#print(xy)

#print(centroids[1])
cx = round(centroids[1][0])
cy = round(centroids[1][1])
#cx = centroids[1][0]
#cy = centroids[1][1]
print(cx)
print(cy)

distance = [0 for i in range(0, len(xy))]
for i in range(0, len(xy)):
    distance[i] = math.sqrt(((x[i]-cx)**2) + ((y[i]-cy)**2))


print("X Values")
print(x)
print("Y Values")
print(y)
print("Distance")
print(distance)

max_distance = max(distance)
max_location = distance.index(max(distance))
print("Max Distance")
print(max_distance)
print("Max Location")
print(max_location)
#print(distance)

angle = [0 for i in range(0, len(xy))] #for now it does not have 0 to 360 degrees
for i in range(0, len(angle)):
    angle[i] = i

#print(angle)

for i in range(0, len(xy)):
    x[i] = x[i] - cx

for i in range(0, len(xy)):
    y[i] = y[i] - cy

print("X-Cx")
print(x)
print("Y-Cy")
print(y)

angles = [0 for i in range(0, len(xy))]

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

#print(angles)
d = np.array([angles])
print(d)
#n_angles = preprocessing.normalize(d)
#print(n_angles)
g = angles[max_location]
print("Angle at max location")
print(g)
for i in range(0, len(angles)):
    angles[i] = angles[i] - g
    if angles[i] < 0:
        angles[i] = angles[i] + 360

d = np.array([angles])
print(d)

xy_pairs = list(zip(angles, distance))
print("X AND Y PAIRS")
print(xy_pairs)
xy_pairs.sort()
print("X AND Y Sorted")
print(xy_pairs)
print(xy_pairs[0][0])
print(xy_pairs[1][0])

new_x = [0 for i in range(0, len(angles))]
new_y = [0 for i in range(0, len(angles))]
for i in range(0, len(contours[0])):
    new_x[i] = xy_pairs[i][0]
for i in range(0, len(contours[0])):
    new_y[i] = xy_pairs[i][1]

#y has the distances
#s[l] = new_y
#
h = [-1/4, 1/2, -1/4]
g = [1/4, 1/2, 1/4]
print(h)
print(g)
a_l = np.convolve(new_y, h, 'valid') #high
d_l = np.convolve(new_y, g, 'valid') #low

a_l_abs = [abs(l) for l in a_l]
d_l_abs = [abs(h) for h in d_l]

sum_a_l = sum(a_l_abs)
sum_d_l = sum(d_l_abs)
print('Sum of |d[l]|: ', sum_d_l, 'Sum of |a[l]|: ', sum_a_l)
beta = sum_a_l/sum_d_l
print('Beta: ', beta)

beta_norm = (beta-8)/(27-8)
print(beta_norm)

#new = [0 for i in range(0, len(angles))]
#for i in range(len(new_y)):
    #if(i < 29):
        #new[i] = new_y[i] - d_l[i]

#print(new)

plot1 = plt.figure(1)
plt.plot(a_l)
plot2 = plt.figure(2)
plt.plot(d_l)

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
#cv2.imshow("fk", f_k_x_y)
#cv2.imshow("g", g_x_y)
pi = 0
b, a = f_k_x_y.shape
for x in range(0, b):
    for y in range(0, a):
        #qprint(g_x_y[x][y])
        if g_x_y[x][y] > s_3:
            if f_k_x_y[x][y] > 0:
                pi_img[x][y] = g_x_y[x][y]
                pi = pi + 1

print('Set Cardinality pi: ', pi)

o = pi/t
print("Intesity o: ", o)

cv2.imshow("pi", pi_img)
cv2.imshow("t", t_img)
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
cnt = contours[0]
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img = cv2.drawContours(img, [box], 0, (155, 0, 0), 1)

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
cv2.imshow("Done", img)

plot3 = plt.figure(3)
plt.plot(new_x, new_y)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
