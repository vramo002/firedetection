import cv2
import numpy as np
import math

fileName = '2021-08-27_08_54_59.avi' #'2021-09-10_08_53_35.avi'
cap = cv2.VideoCapture(fileName)

lower_red = np.array([0, 100, 100])
upper_red = np.array([0, 255, 255])

fps = cap.get(cv2.CAP_PROP_FPS)
w1 = cap.get(3)
h1 = cap.get(4)
centerx = w1/2
centery = h1/2
roistartw = cap.get(3) / 10
roistarth = cap.get(4) / 10
print(str(round(w1/2) )+ " " + str(round(h1/2)))

print(cap.get(3))
print(cap.get(4))
pixel_len = 1.074285712 #0.067159167 #cm and if object is 1 meter
N = 0
E = 0
while True:
    ret, frame = cap.read()
    #frame = frame[int(roistarth) * 4:int(h1) - int(roistarth) * 4, int(roistartw) * 4:int(w1) - int(roistartw) * 4]
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame1, lower_red, upper_red)
    res = cv2.bitwise_and(frame1, frame1, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    cv2.imshow("og", frame)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #ret, frame = cv2.threshold(frame, 255 * .6, 255, cv2.THRESH_BINARY)
    #print(cv2.countNonZero(frame))
    #cv2.imshow("binary", frame)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, None, None, None, 8, cv2.CV_32S)
    #cv2.imshow("connected", frame)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.imshow("countor", frame)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        #if(w > 50 and h > 50):
        #print("x " + str(x))
        #print("y " + str(y))
        print("w " + str(w))
        print("h " + str(h))
        distance_w = w * pixel_len
        distance_h = h * pixel_len
        print('w ' + str(distance_w) + ' ' + 'h ' + str(distance_h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cx = round(centroids[1][0])
        cy = round(centroids[1][1])
        cv2.line(frame, (round(centerx), round(centery)), (cx, cy), color=(255, 0, 0), thickness=5)
        distance = math.sqrt(((centerx - cx) ** 2) + ((centery - cy) ** 2))
        print(distance*pixel_len)
        if cy < 160 & cx < 240:
            N = abs(cy-centery)/100
            E = -abs(cx-centerx)/100
        elif cy < 160 & cx >= 240:
            N = abs(cy - centery)/100
            E = abs(cx - centerx)/100
        elif cy >= 160 & cx < 240:
            N = -abs(cy - centery)/100
            E = -abs(cx - centerx)/100
        elif cy >= 160 & cx >= 240:
            N = -abs(cy - centery)/100
            E = abs(cx - centerx)/100
        print(str(N) + " " + str(E))
    cv2.imshow("rec", frame)
    frame = cv2.circle(frame, (round(w1/2), round(h1/2)), radius=0, color=(255, 0, 0), thickness=5)
    cv2.imshow("point", frame)
    cv2.waitKey(int((1 / int(fps)) * 1000))


cv2.waitKey(0)
cv2.destroyAllWindows()
