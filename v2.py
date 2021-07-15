import numpy as np
import cv2
import imutils
import time
from datetime import datetime
import os

now = datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H_%M_%S")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID'))
writer = cv2.VideoWriter(time_stamp + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (int(cap.get(3)), int(cap.get(4))))

fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(3)
h = cap.get(4)
roistartw = cap.get(3)/10
roistarth = cap.get(4)/10

lower_red = np.array([0,100,100])
upper_red = np.array([179,255,255])

counter = 10

while(True):

    ret, frame = cap.read()
    writer.write(frame)

    
    if not ret:
        break
    cv2.imshow('original', frame)
    #frame = imutils.resize(frame, width=320)
    #frame3 = cv2.rectangle(frame, (int(roistartw)*3,int(roistarth)*3), (int(w)-int(roistartw)*3,int(h)-int(roistarth)*3), (0,255,0), 2)
    #cv2.imshow('roibox', frame3)
    frame = frame[int(roistarth)*3:int(h)-int(roistarth)*3,int(roistartw)*3:int(w)-int(roistartw)*3]
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame1, lower_red, upper_red)
    res = cv2.bitwise_and(frame1, frame1, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) != 0:
        counter = counter - 1
        if counter == 0:
            print("FIRE")
    #if cv2.countNonZero(gray) == 0:
        #print("NO FIRE")
    cv2.imshow('ROI', frame)
   # cv2.imshow('mask', mask)
   # cv2.imshow('res', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(int((1/int(fps))*1000))
    
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break

cap.release()
cv2.destroyAllWindows()
