import numpy as np
import cv2
import imutils

fileName = 'persons_1.mp4'  # change the file name if needed

cap = cv2.VideoCapture(fileName)  # load the video
print(cap.get(3))
print(cap.get(4))
print(cap.get(cv2.CAP_PROP_FPS))
fps = cap.get(cv2.CAP_PROP_FPS)
print(int((1/int(fps))*1000))

while (cap.isOpened()):  # play the video by reading frame by frame
    ret, frame = cap.read()
    if ret == True:
        # optional: do some image processing here
        frame = imutils.resize(frame, width=320)
        #print(frame.shape)
        cv2.imshow('video', frame)  # show the video
        cv2.waitKey(int((1/int(fps))*1000))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
