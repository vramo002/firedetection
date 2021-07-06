import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture(1)

fps = cap.get(cv2.CAP_PROP_FPS)

lower_red = np.array([0,100,100])
upper_red = np.array([179,255,255])

while(cap.isOpened()):

    ret, frame = cap.read()
    
    if not ret:
        break

    frame = imutils.resize(frame, width=320)
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame1, lower_red, upper_red)
    res = cv2.bitwise_and(frame1, frame1, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) != 0:
        print("FIRE")
    if cv2.countNonZero(gray) == 0:
        print("NO FIRE")
    cv2.imshow('original', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.waitKey(int((1/int(fps))*1000))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
