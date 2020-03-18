import cv2
import numpy as np 

image1 = []

cap = cv2.VideoCapture("v01.mp4")
while not cap.isOpened():
    cap = cv2.VideoCapture("./v01.mp4")
    cv2.waitKey(1000)
    print ("wait for header")

pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

while True:
    bool, frame1 = cap.read()
    if flag:
        np_frame = cv2.imread('v01.mp4', frame1)
        image1.append(np_frame)

        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    else:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
        print ("frame not ready")
        cv2.waitKey(1000)

    if cv2.waitKey(10) == 27:
        break
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        break

all_frames = np.array(image1)