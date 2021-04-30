import cv2
import imutils

vidcap = cv2.VideoCapture('5_3.mp4')
success, image = vidcap.read()
count = 0
while success:
    image = imutils.resize(image, width=320)
    cv2.imwrite("frame5_3#%d.jpg" % count, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
